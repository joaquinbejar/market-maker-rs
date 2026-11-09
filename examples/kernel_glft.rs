//! Financial kernel driven by GLFT quotes.
//!
//! Quotes of Guéant, Lehalle and Fernandez-Tapia (arXiv:1105.3115,
//! Theorem 1) through `GlftQuoter`, with prices in ticks around a mid of
//! 1000 and the inventory bounded by the model's `Q` lots and the kernel's
//! position limit. A one-sided quote at the inventory bound becomes a
//! `QuoteError`, recorded as a hold.
//!
//! Run with: `cargo run --example kernel_glft --features kernel`

use market_maker_rs::backtest::MarketTick;
use market_maker_rs::backtest::kernel::{
    FinancialKernel, GlftQuoter, KernelAction, KernelConfig, KernelDataset, QuoteAdapter,
};
use market_maker_rs::dec;
use market_maker_rs::prelude::*;
use market_maker_rs::strategy::glft::GLFTConfig;
use rust_decimal::prelude::FromPrimitive;

const SECONDS_PER_YEAR: f64 = 31_536_000.0;

fn main() -> MMResult<()> {
    let glft = GLFTConfig::new(
        dec!(0.01),
        dec!(0.3),
        dec!(0.9),
        30,
        dec!(1),
        600_000,
        dec!(0),
    )?;
    let sigma = Decimal::from_f64(0.3 * SECONDS_PER_YEAR.sqrt())
        .ok_or_else(|| MMError::NumericalError("sigma".to_string()))?;

    let cfg = KernelConfig::new(dec!(1), dec!(1), dec!(5000), dec!(5), 10_000)
        .with_initial_capital(dec!(1000000))
        .with_limits(RiskLimits::new(dec!(30), Decimal::MAX, dec!(0))?);

    // A random walk of the mid, one row per second for 600 seconds.
    let mut state: u64 = 11;
    let mut mid: i64 = 1_000;
    let mut ticks = Vec::new();
    for i in 0..600u64 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        mid += i64::try_from((state >> 60) % 13).unwrap_or(0) - 6;
        ticks.push(MarketTick::new(
            1_000 * i,
            Decimal::from(mid - 1),
            dec!(5),
            Decimal::from(mid + 1),
            dec!(5),
        ));
    }
    let data = KernelDataset::from_ticks(&ticks, &cfg)?;
    let kernel: FinancialKernel<'_> = FinancialKernel::new(&cfg, &data)?;
    let quoter = GlftQuoter::new(glft, sigma, dec!(1), 0);
    let mut adapter = QuoteAdapter::new(quoter, &kernel, dec!(1))?;

    let mut s = kernel.reset(1);
    let (mut quotes, mut errors) = (0u64, 0u64);
    while s.status().is_running() {
        let obs = kernel.observe(&s);
        let action = adapter.act(&kernel, &obs);
        match action {
            KernelAction::Quote { .. } | KernelAction::OneSided { .. } => quotes += 1,
            KernelAction::QuoteError(_) => errors += 1,
            KernelAction::Hold | KernelAction::Cancel => {}
            // The adapter quotes without an expiry.
            _ => {}
        }
        kernel.step(&mut s, action);
    }
    let summary = kernel.venue_summary(&s)?;
    println!("=== Financial kernel: GLFT ===");
    println!("status:       {:?}", s.status());
    println!("quotes:       {quotes} (errors recorded as holds: {errors})");
    println!("fills:        {}", s.counters().venue_fills);
    println!("position:     {}", summary.position);
    println!("net pnl:      {}", summary.net_pnl);
    Ok(())
}
