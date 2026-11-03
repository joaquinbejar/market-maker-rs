//! Financial kernel driven by Avellaneda-Stoikov quotes.
//!
//! Builds a seeded random walk on a 0.01 tick grid, validates it into a
//! `KernelDataset`, and steps a `FinancialKernel` with the quotes of
//! `AvellanedaQuoter` (Avellaneda and Stoikov, 2008) through a
//! `QuoteAdapter`, as an external RL trainer would step it with its own
//! policy.
//!
//! Run with: `cargo run --example kernel_avellaneda --features kernel`

use market_maker_rs::backtest::MarketTick;
use market_maker_rs::backtest::kernel::{
    AvellanedaQuoter, FinancialKernel, KernelConfig, KernelDataset, KernelLatency, QuoteAdapter,
    encode, fingerprint,
};
use market_maker_rs::dec;
use market_maker_rs::prelude::*;

fn main() -> MMResult<()> {
    let cfg = KernelConfig::new(dec!(0.01), dec!(0.1), dec!(1000), dec!(10), 10_000)
        .with_initial_capital(dec!(100000))
        .with_fee_rate(dec!(0.0002))
        .with_latency(
            KernelLatency::zero(1_000_000)
                .with_delays(1, 2, 2, 1)
                .with_jitter(1),
        );

    // Seeded random walk around 100.00, one row every 100 ms.
    let mut state: u64 = 42;
    let mut mid: i64 = 10_000;
    let mut ticks = Vec::new();
    for i in 0..2_000u64 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        mid += i64::try_from((state >> 60) % 7).unwrap_or(0) - 3;
        let bid = Decimal::from(mid - 1) / dec!(100);
        let ask = Decimal::from(mid + 1) / dec!(100);
        ticks.push(MarketTick::new(1_000 + 100 * i, bid, dec!(5), ask, dec!(5)));
    }
    let data = KernelDataset::from_ticks(&ticks, &cfg)?;
    let kernel: FinancialKernel<'_> = FinancialKernel::new(&cfg, &data)?;

    let quoter = AvellanedaQuoter {
        risk_aversion: dec!(0.1),
        volatility: dec!(0.2),
        order_intensity: dec!(50),
        session_start_ms: 1_000,
        horizon_ms: 100 * 2_000,
        size: dec!(0.5),
    };
    let mut adapter = QuoteAdapter::new(quoter, &kernel, dec!(0.1))?;

    let mut s = kernel.reset(7);
    let mut decisions = 0u64;
    while s.status().is_running() {
        let obs = kernel.observe(&s);
        let action = adapter.act(&kernel, &obs);
        kernel.step(&mut s, action);
        decisions += 1;
    }

    let summary = kernel.venue_summary(&s)?;
    println!("=== Financial kernel: Avellaneda-Stoikov ===");
    println!("status:        {:?}", s.status());
    println!("decisions:     {decisions}");
    println!("fills:         {}", s.counters().venue_fills);
    println!("position:      {}", summary.position);
    println!("net pnl:       {}", summary.net_pnl);
    println!("fees:          {}", summary.total_fees);
    println!("max drawdown:  {}", summary.max_drawdown);
    println!("state bytes:   {}", encode(&s).len());
    println!("fingerprint:   {:032x}", fingerprint(&s));
    Ok(())
}
