//! What a step costs in Rust, directly and through the worker's channels, and what the
//! worker's episode recorder holds at the end of an episode.
//!
//! `cargo run --release --example step_cost`. Synthetic rows. It measures one machine, once.

use std::time::Instant;

use market_maker_rs::Decimal;
use market_maker_rs::backtest::kernel::{
    ActionGrid, FillRec, KernelConfig, KernelDataset, KernelTick,
};
use market_maker_rs::rl::{EpisodeRecorder, ObservationScales, RlAction, RlConfig, RlEnv};
use mm_rl_bridge::worker::{BridgeConfig, DLV, MSG, OBS, ORD, Worker};

const ROWS: u64 = 200_000;

fn rows() -> Vec<KernelTick> {
    (0..ROWS)
        .map(|i| {
            let bid = 190 + (i * 7) % 13;
            KernelTick {
                ts_ns: (1000 + i * 25) * 1_000_000,
                bid,
                bid_lots: 4,
                ask: bid + 2,
                ask_lots: 3,
            }
        })
        .collect()
}

fn action(step: u64) -> RlAction {
    if step.is_multiple_of(3) {
        RlAction::Quote {
            bid: 0,
            ask: 0,
            bid_size: 0,
            ask_size: 0,
        }
    } else {
        RlAction::Hold
    }
}

fn config() -> BridgeConfig {
    BridgeConfig::from_json(&format!(
        r#"{{"tick_size": "0.5", "lot_size": "1", "max_price": "1000", "max_order_size": "10",
            "max_steps": {ROWS}, "fee_rate": "0.001", "initial_capital": "0",
            "latency": {{"quantum_ns": 1000000, "observation_q": 0, "submit_q": 0, "cancel_q": 0,
                         "ack_q": 0, "jitter_q": 0}},
            "fill_model": {{"kind": "immediate"}}, "limits": null,
            "bid_offsets": [1], "ask_offsets": [1], "sizes": [1],
            "scales": {{"spread_ticks": "10", "mid_change_ticks": "4", "row_gap_ms": "4000",
                        "size_lots": "10", "position": "4", "entry_gap": "2"}},
            "f32_unit": "1", "invalid_action": "mask", "invalid_action_penalty": "0",
            "terminal": "mark"}}"#
    ))
    .expect("a valid configuration")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let kernel = KernelConfig::new(
        Decimal::new(5, 1),
        Decimal::ONE,
        Decimal::from(1000),
        Decimal::TEN,
        ROWS,
    )
    .with_fee_rate(Decimal::new(1, 3));
    let data = KernelDataset::from_kernel_ticks(rows(), &kernel)?;
    let scales = ObservationScales {
        spread_ticks: Decimal::from(10),
        mid_change_ticks: Decimal::from(4),
        row_gap_ms: Decimal::from(4000),
        size_lots: Decimal::from(10),
        position: Decimal::from(4),
        entry_gap: Decimal::from(2),
    };
    let env: RlEnv<'_, 1, 1, ORD, MSG, DLV, OBS> = RlEnv::new(
        &kernel,
        &data,
        ActionGrid::new([1], [1], [1])?,
        RlConfig::new(scales, Decimal::from(1)),
    )?;
    let mut state = env.reset(1)?;
    let start = Instant::now();
    let mut steps = 0_u64;
    while !state.is_done() {
        env.step(&mut state, action(steps))?;
        steps += 1;
    }
    let direct = start.elapsed().as_nanos() / u128::from(steps);

    // The worker records every episode, as `attribution` needs: one row index per completed
    // row and one record per fill. A `Vec` may hold up to twice what it has grown to.
    let mut recorder = EpisodeRecorder::default();
    let mut state = env.reset_with(1, &mut recorder)?;
    let start = Instant::now();
    let mut recorded = 0_u64;
    while !state.is_done() {
        env.step_with(&mut state, action(recorded), &mut recorder)?;
        recorded += 1;
    }
    let with_recorder = start.elapsed().as_nanos() / u128::from(recorded);
    let records = env.records(&state, &recorder)?;
    let rows_bytes = records.marks.len() * std::mem::size_of::<u64>();
    let fills_bytes = records.fills.len() * std::mem::size_of::<FillRec>();
    let held = rows_bytes + fills_bytes;

    let mut worker = Worker::spawn(config(), rows(), None)?;
    worker.reset(1)?;
    let start = Instant::now();
    let mut through = 0_u64;
    while through < steps {
        worker.step(action(through))?;
        through += 1;
    }
    let channels = start.elapsed().as_nanos() / u128::from(through);
    worker.close()?;
    println!(
        "direct {direct} ns per step, through the worker {channels} ns per step, {steps} steps"
    );
    println!(
        "with the recorder {with_recorder} ns per step; at the end it holds {} rows ({rows_bytes} \
         bytes) and {} fills ({fills_bytes} bytes, {} each): {held} bytes, {:.1} per step, up to \
         twice that allocated",
        records.marks.len(),
        records.fills.len(),
        std::mem::size_of::<FillRec>(),
        held as f64 / recorded as f64,
    );
    Ok(())
}
