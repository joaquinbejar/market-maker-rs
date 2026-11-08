//! Exports `GreeksRiskManager`'s post-or-hold decision for every inventory state of #111's grids.
//!
//! Usage: `qrh_gate_table <example_1|example_2> <scale> <out>`. Each output line holds one state,
//! in the grid's row-major order, as `allowed scaled`: two bitmasks over the moves (asset j's bid
//! is bit 2j, its ask bit 2j + 1). Only `Allowed` posts; the circuit breaker, which depends on the
//! path, is reset at each state so the table is the limit test alone.

use market_maker_rs::Decimal;
use market_maker_rs::options::{
    AutoHedgerConfig, GreeksLimits, GreeksRiskManager, OrderDecision, PositionGreeks,
};
use rust_decimal_macros::dec;
use std::io::Write;
use std::str::FromStr;

const LATTICE: Decimal = dec!(0.002);
const JOINT_SECOND_DELTA: Decimal = dec!(-0.028);

struct Asset {
    delta: Decimal,
    size: Decimal,
}

fn manager(limit: Decimal) -> GreeksRiskManager {
    let huge = dec!(1000000000000);
    GreeksRiskManager::new(
        "QRH",
        GreeksLimits::new(limit, huge, huge, -huge, limit),
        AutoHedgerConfig::default(),
    )
}

fn unit(delta: Decimal) -> PositionGreeks {
    PositionGreeks::new(delta, dec!(0), dec!(0), dec!(0), dec!(0))
}

/// One state's masks: `moves[i]` is whether move i stays on the grid.
fn decide(
    gate: &mut GreeksRiskManager,
    booked: &mut Decimal,
    risk: Decimal,
    assets: &[Asset],
    moves: &[bool],
) -> (u16, u16) {
    gate.update_on_fill(&unit(dec!(1)), risk - *booked, 0)
        .expect("booking a state cannot overflow");
    *booked = risk;
    gate.reset_circuit_breaker();
    let (mut allowed, mut scaled) = (0, 0);
    for (j, asset) in assets.iter().enumerate() {
        for (side, sign) in [dec!(1), dec!(-1)].into_iter().enumerate() {
            let bit = 2 * j + side;
            if !moves[bit] {
                continue;
            }
            let decision = gate
                .check_order(&unit(asset.delta), sign * asset.size)
                .expect("a check on these values cannot overflow");
            match decision {
                OrderDecision::Allowed => allowed |= 1 << bit,
                OrderDecision::Scaled { .. } => scaled |= 1 << bit,
                OrderDecision::Rejected { .. } => {}
            }
        }
    }
    (allowed, scaled)
}

fn example_1(limit: Decimal, second_delta: Decimal, reversed: bool) -> Vec<(u16, u16)> {
    let assets = [
        Asset {
            delta: dec!(1),
            size: dec!(1),
        },
        Asset {
            delta: second_delta,
            size: dec!(20),
        },
    ];
    let states: Vec<(i64, i64)> = (-15..=15)
        .flat_map(|i| (-15..=15).map(move |j| (i, j)))
        .collect();
    let positions: Vec<usize> = ordered(states.len(), reversed);
    let mut gate = manager(limit);
    let mut booked = dec!(0);
    let mut out = vec![(0, 0); states.len()];
    for k in positions {
        let (i, j) = states[k];
        let risk = Decimal::from(i) * assets[0].delta
            + Decimal::from(j) * assets[1].size * assets[1].delta;
        let moves = [i < 15, i > -15, j < 15, j > -15];
        out[k] = decide(&mut gate, &mut booked, risk, &assets, &moves);
    }
    out
}

fn example_2(limit: Decimal, reversed: bool) -> Vec<(u16, u16)> {
    let assets: Vec<Asset> = [dec!(0.533), dec!(0.134), dec!(-0.014), dec!(-0.013)]
        .into_iter()
        .map(|delta| Asset {
            delta,
            size: dec!(2),
        })
        .collect();
    let mut gate = manager(limit);
    let mut booked = dec!(0);
    let mut out = vec![(0, 0); 10001];
    for k in ordered(10001, reversed) {
        let level = k as i64 - 5000;
        let risk = Decimal::from(level) * LATTICE;
        let moves: Vec<bool> = assets
            .iter()
            .flat_map(|a| {
                let step = a.size * a.delta / LATTICE;
                let step = i64::try_from(step).expect("the step is on the lattice");
                [(level + step).abs() <= 5000, (level - step).abs() <= 5000]
            })
            .collect();
        out[k] = decide(&mut gate, &mut booked, risk, &assets, &moves);
    }
    out
}

fn ordered(n: usize, reversed: bool) -> Vec<usize> {
    if reversed {
        (0..n).rev().collect()
    } else {
        (0..n).collect()
    }
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let scale = Decimal::from_str(&args[2]).expect("scale is a decimal");
    let table = match args[1].as_str() {
        "example_1" => example_1(dec!(23.4) * scale, JOINT_SECOND_DELTA, false),
        "example_1_decoupled" => example_1(dec!(15) * scale, dec!(0), false),
        "example_2" => example_2(dec!(10) * scale, false),
        other => panic!("unknown example {other}"),
    };
    let mut out = std::io::BufWriter::new(std::fs::File::create(&args[3])?);
    for (allowed, scaled) in table {
        writeln!(out, "{allowed} {scaled}")?;
    }
    out.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_does_not_depend_on_the_visiting_order() {
        assert_eq!(example_2(dec!(5), false), example_2(dec!(5), true));
        assert_eq!(
            example_1(dec!(11.7), JOINT_SECOND_DELTA, false),
            example_1(dec!(11.7), JOINT_SECOND_DELTA, true)
        );
    }

    #[test]
    fn a_reducing_order_at_the_limit_gets_the_decision_after_the_reset() {
        let mut gate = manager(dec!(10));
        gate.update_on_fill(&unit(dec!(1)), dec!(10), 0).unwrap();
        gate.reset_circuit_breaker();
        let assets = [Asset {
            delta: dec!(1),
            size: dec!(1),
        }];
        let decision = gate.check_order(&unit(dec!(1)), dec!(-1)).unwrap();
        assert!(matches!(decision, OrderDecision::Allowed));
        let mut booked = dec!(10);
        let (allowed, _) = decide(&mut gate, &mut booked, dec!(10), &assets, &[true, true]);
        assert_eq!(allowed & 0b10, 0b10);
        assert_eq!(allowed & 0b01, 0);
    }

    #[test]
    fn at_scale_one_example_1_posts_every_move_on_the_grid() {
        let table = example_1(dec!(23.4), JOINT_SECOND_DELTA, false);
        assert_eq!(table[480], (0b1111, 0));
        assert!(table.iter().all(|&(_, scaled)| scaled == 0));
    }

    #[test]
    fn the_decoupled_table_never_binds_beyond_the_box() {
        let table = example_1(dec!(15), dec!(0), false);
        assert_eq!(table[480], (0b1111, 0));
        assert!(table.iter().all(|&(_, scaled)| scaled == 0));
    }
}
