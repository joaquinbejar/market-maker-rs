//! The feature vector, from what the agent knows.

use rust_decimal::prelude::ToPrimitive;

use crate::Decimal;
use crate::backtest::kernel::Observation;
use crate::types::error::{MMError, MMResult};

use super::spec::{FEATURE_COUNT, FeatureId, FeatureSpec, HEDGE_FEATURE_COUNT, ObservationSpace};

/// Nanoseconds as a `Decimal` count of milliseconds has this scale.
const NS_SCALE: u32 = 6;

/// What the agent saw at its previous decision: the only memory of the observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct PreviousRow {
    /// `bid + ask` of that decision's observed row, in ticks: twice its mid.
    pub mid2: u64,
    /// That row's exchange time, in nanoseconds.
    pub ts_ns: u64,
}

impl PreviousRow {
    /// A previous row of `mid2` half ticks at `ts_ns`.
    #[must_use]
    pub fn new(mid2: u64, ts_ns: u64) -> Self {
        Self { mid2, ts_ns }
    }

    /// What a decision on `obs` leaves for the next one, as an environment keeps it: its
    /// observed row, or `None` with no row, in which case the memory stays as it was.
    #[must_use]
    pub fn of(obs: &Observation) -> Option<Self> {
        obs.row?;
        Some(Self::new(obs.bid.checked_add(obs.ask)?, obs.ts_ns))
    }
}

/// The observation at a decision.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct FeatureVector {
    /// The last row delivered to the agent, which these values are as of.
    pub row: Option<u64>,
    /// Its exchange time, in nanoseconds (0 if none).
    pub ts_ns: u64,
    /// Each feature's value before normalization, in its declared unit, and `None`
    /// when it has no value at this decision. In the order of [`FeatureId::ALL`].
    pub raw: [Option<Decimal>; FEATURE_COUNT],
    /// `raw / divisor`, clipped to `[-1, 1]`, and 0 for a feature with no value.
    /// Derived from [`FeatureVector::raw`].
    pub values: [f32; FEATURE_COUNT],
}

/// The hedge features at a decision, with a hedge lane.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct HedgeFeatureVector {
    /// The hedge row the agent sees, which the row values are as of.
    pub row: Option<u64>,
    /// Each hedge feature's value before normalization, in its declared unit, and `None`
    /// when it has no value. In the order of [`FeatureId::HEDGE`].
    pub raw: [Option<Decimal>; HEDGE_FEATURE_COUNT],
    /// `raw / divisor`, clipped to `[-1, 1]`, and 0 for a feature with no value.
    pub values: [f32; HEDGE_FEATURE_COUNT],
}

impl HedgeFeatureVector {
    pub(super) fn new(
        row: Option<u64>,
        raw: [Option<Decimal>; HEDGE_FEATURE_COUNT],
        specs: &[FeatureSpec],
    ) -> MMResult<Self> {
        let mut values = [0.0_f32; HEDGE_FEATURE_COUNT];
        for ((value, exact), spec) in values.iter_mut().zip(&raw).zip(specs) {
            if let Some(exact) = exact {
                *value = exact
                    .checked_div(spec.divisor)
                    .ok_or_else(|| numeric("normalized hedge feature"))?
                    .clamp(Decimal::NEGATIVE_ONE, Decimal::ONE)
                    .to_f32()
                    .ok_or_else(|| numeric("normalized hedge feature"))?;
            }
        }
        Ok(Self { row, raw, values })
    }
}

fn numeric(what: &str) -> MMError {
    MMError::NumericalError(format!("{what} does not convert"))
}

/// The feature's value on its own row of the table in [`ObservationSpace`].
fn raw_value(
    id: FeatureId,
    obs: &Observation,
    previous: Option<PreviousRow>,
    mid: Option<Decimal>,
    max_steps: u64,
) -> MMResult<Option<Decimal>> {
    let observed = obs.row.is_some();
    let on_row = |value: u64| observed.then(|| Decimal::from(value));
    Ok(match id {
        FeatureId::Spread => obs.ask.checked_sub(obs.bid).and_then(on_row),
        FeatureId::MidChange => match previous {
            Some(row) => {
                let halves = i128::from(obs.bid)
                    .checked_add(i128::from(obs.ask))
                    .and_then(|mid2| mid2.checked_sub(i128::from(row.mid2)))
                    .and_then(|d| d.checked_mul(5))
                    .ok_or_else(|| numeric("mid change"))?;
                Some(
                    Decimal::try_from_i128_with_scale(halves, 1)
                        .map_err(|_| numeric("mid change"))?,
                )
            }
            None => None,
        },
        FeatureId::RowGap => match previous {
            Some(row) => {
                let ns = obs
                    .ts_ns
                    .checked_sub(row.ts_ns)
                    .ok_or_else(|| numeric("row gap"))?;
                Some(
                    Decimal::try_from_i128_with_scale(i128::from(ns), NS_SCALE)
                        .map_err(|_| numeric("row gap"))?,
                )
            }
            None => None,
        },
        FeatureId::BidSize => on_row(obs.bid_lots),
        FeatureId::AskSize => on_row(obs.ask_lots),
        FeatureId::Imbalance => {
            let total = obs.bid_lots.checked_add(obs.ask_lots);
            match total {
                Some(total) if total > 0 => {
                    let difference = Decimal::from(obs.bid_lots)
                        .checked_sub(Decimal::from(obs.ask_lots))
                        .ok_or_else(|| numeric("imbalance"))?;
                    Some(
                        difference
                            .checked_div(Decimal::from(total))
                            .ok_or_else(|| numeric("imbalance"))?,
                    )
                }
                _ => None,
            }
        }
        FeatureId::Position => Some(obs.position),
        FeatureId::EntryGap => match mid {
            Some(mid) if !obs.position.is_zero() => Some(
                mid.checked_sub(obs.avg_entry)
                    .ok_or_else(|| numeric("entry gap"))?,
            ),
            _ => None,
        },
        FeatureId::OpenOrders => Some(Decimal::from(obs.open_orders)),
        FeatureId::AwaitingCancel => Some(Decimal::from(obs.awaiting_cancel_outcome)),
        FeatureId::StepsLeft => Some(Decimal::from(max_steps.saturating_sub(obs.step))),
        // The hedge features are in the hedge vector.
        FeatureId::HedgeSpread
        | FeatureId::HedgeBasis
        | FeatureId::HedgeRowAge
        | FeatureId::HedgePosition
        | FeatureId::HedgeInFlight => None,
    })
}

/// The feature vector of `obs`, from explicit inputs and no hidden state: `previous` is
/// what the previous decision left ([`PreviousRow::of`] the observation it decided on, or
/// the memory before it when it saw no row), `tick_size` values one tick, and `max_steps`
/// is the specification's. It is what [`RlEnv::observe`](crate::rl::RlEnv::observe) returns
/// for the same inputs.
///
/// # Errors
///
/// Returns [`MMError::NumericalError`] if a value does not convert.
pub fn observation_features(
    space: &ObservationSpace,
    obs: &Observation,
    previous: Option<PreviousRow>,
    tick_size: Decimal,
    max_steps: u64,
) -> MMResult<FeatureVector> {
    let mid = match obs.row {
        Some(_) => {
            let mid2 = obs
                .bid
                .checked_add(obs.ask)
                .ok_or_else(|| numeric("observed mid"))?;
            let twice = Decimal::from(mid2)
                .checked_mul(tick_size)
                .ok_or_else(|| numeric("observed mid"))?;
            Some(
                twice
                    .checked_mul(Decimal::new(5, 1))
                    .ok_or_else(|| numeric("observed mid"))?,
            )
        }
        None => None,
    };
    feature_vector(space, obs, previous, mid, max_steps)
}

/// The feature vector of `obs`. `mid` is the mid of the observed row, if any.
pub(super) fn feature_vector(
    space: &ObservationSpace,
    obs: &Observation,
    previous: Option<PreviousRow>,
    mid: Option<Decimal>,
    max_steps: u64,
) -> MMResult<FeatureVector> {
    let mut raw = [None; FEATURE_COUNT];
    let mut values = [0.0_f32; FEATURE_COUNT];
    for ((slot, value), spec) in raw.iter_mut().zip(values.iter_mut()).zip(&space.features) {
        *slot = raw_value(spec.id, obs, previous, mid, max_steps)?;
        if let Some(exact) = *slot {
            let scaled = exact
                .checked_div(spec.divisor)
                .ok_or_else(|| numeric("normalized feature"))?;
            *value = scaled
                .clamp(Decimal::NEGATIVE_ONE, Decimal::ONE)
                .to_f32()
                .ok_or_else(|| numeric("normalized feature"))?;
        }
    }
    Ok(FeatureVector {
        row: obs.row,
        ts_ns: obs.ts_ns,
        raw,
        values,
    })
}
