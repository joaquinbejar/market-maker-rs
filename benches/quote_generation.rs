//! Benchmarks for quote generation.
//!
//! Measures the real quoting kernels with fixed inputs over a small inventory
//! grid (-10, 0, 10). Inputs and outputs go through `black_box` so the
//! compiler cannot remove the work. Results depend on machine, toolchain and
//! build profile; record them alongside the commit SHA when comparing runs.
//!
//! Run with `cargo bench --bench quote_generation`.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use market_maker_rs::dec;
use market_maker_rs::strategy::avellaneda_stoikov::{
    calculate_optimal_quotes, calculate_optimal_spread, calculate_reservation_price,
};
use market_maker_rs::strategy::glft::{GLFTConfig, GLFTStrategy};

const MID: &str = "100.0";
const INVENTORIES: [i64; 3] = [-10, 0, 10];
const TIME_TO_TERMINAL_MS: u64 = 3_600_000;

fn avellaneda_stoikov(c: &mut Criterion) {
    let mid = MID.parse().expect("valid mid");
    let gamma = dec!(0.1);
    let sigma = dec!(0.2);
    let k = dec!(1.5);

    let mut group = c.benchmark_group("avellaneda_stoikov");
    for inv in INVENTORIES {
        let q = market_maker_rs::Decimal::from(inv);
        group.bench_with_input(BenchmarkId::new("reservation_price", inv), &q, |b, q| {
            b.iter(|| {
                calculate_reservation_price(
                    black_box(mid),
                    black_box(*q),
                    black_box(gamma),
                    black_box(sigma),
                    black_box(TIME_TO_TERMINAL_MS),
                )
            })
        });
        group.bench_with_input(BenchmarkId::new("optimal_quotes", inv), &q, |b, q| {
            b.iter(|| {
                calculate_optimal_quotes(
                    black_box(mid),
                    black_box(*q),
                    black_box(gamma),
                    black_box(sigma),
                    black_box(TIME_TO_TERMINAL_MS),
                    black_box(k),
                )
            })
        });
    }
    group.bench_function("optimal_spread", |b| {
        b.iter(|| {
            calculate_optimal_spread(
                black_box(gamma),
                black_box(sigma),
                black_box(TIME_TO_TERMINAL_MS),
                black_box(k),
            )
        })
    });
    group.finish();
}

fn glft(c: &mut Criterion) {
    let mid = MID.parse().expect("valid mid");
    // Paper parameters (Figure 1), sigma = 0.3 per sqrt(s) expressed per sqrt(year).
    let sigma = dec!(1684.7);
    let mut group = c.benchmark_group("glft");
    for q_max in [30_u32, 200] {
        let config = GLFTConfig::new(
            dec!(0.01),
            dec!(0.3),
            dec!(0.9),
            q_max,
            dec!(1),
            600_000,
            dec!(0),
        )
        .expect("valid GLFT config");
        for inv in INVENTORIES {
            let q = market_maker_rs::Decimal::from(inv);
            group.bench_with_input(
                BenchmarkId::new(format!("optimal_quotes_q{q_max}"), inv),
                &q,
                |b, q| {
                    b.iter(|| {
                        GLFTStrategy::calculate_optimal_quotes(
                            black_box(mid),
                            black_box(*q),
                            black_box(&config),
                            black_box(sigma),
                            black_box(0),
                        )
                    })
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, avellaneda_stoikov, glft);
criterion_main!(benches);
