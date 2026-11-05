//! The Avellaneda-Stoikov baseline quoting from walk-forward calibration.
//!
//! [`ParameterSchedule`] says which parameters hold when. [`CalibratedQuoter`]
//! is a [`BacktestStrategy`] that quotes the validated baseline,
//! [`calculate_optimal_quotes`], from them in the causal replay, so a calibrated
//! and a constant schedule run under the same engine, fees, latency and limits.
//!
//! # Causality
//!
//! A fold's parameters hold only during its test window, and a schedule refuses
//! a record whose training ends after its test window starts. So a tick reads
//! parameters estimated before it. Outside every window, in a fold that could not
//! calibrate, or when a quote cannot be computed, the quoter is silent and cancels
//! what rests, so no new quote comes from an earlier window's parameters.
//!
//! An order already out can still outlive its window. A tick matches resting
//! orders before the quoter hears of it, and under simulated latency a quote from
//! a window's last ticks may activate after it ends, and a cancel takes its own
//! delay. So fills can land in a later window from earlier parameters.
//!
//! # What enters the quotes
//!
//! The baseline takes the absolute volatility and `k`. The intensity scale `A` and
//! the seasonality `s(t)` do not enter its quotes, and the fill model of the
//! replay is its own, so they are not used here. The volatility is absolute at
//! the last training price, so it goes stale if the price moves far from there.
//! The risk aversion and the horizon are the caller's, as nothing here calibrates
//! them. A record with [`CalibrationRecord::open_ms`] set has its volatility per
//! √year of open time, so pass a horizon in open time as well.
//!
//! [`CalibrationRecord::open_ms`]: crate::strategy::walk_forward::CalibrationRecord::open_ms
//!
//! # Example
//!
//! ```
//! use market_maker_rs::backtest::calibrated::{CalibratedQuoter, ParameterSchedule};
//! use market_maker_rs::dec;
//! use market_maker_rs::strategy::walk_forward::{ConstantParameters, TimeWindow};
//!
//! let window = TimeWindow::new(1_000, 2_000).unwrap();
//! let constant = ConstantParameters {
//!     volatility: dec!(2),
//!     intensity_scale_a: dec!(1),
//!     order_intensity_k: dec!(1.5),
//! };
//! let schedule = ParameterSchedule::constant(window, &constant).unwrap();
//! let quoter = CalibratedQuoter::new(schedule, dec!(0.1), 3_600_000, dec!(1)).unwrap();
//! assert_eq!(quoter.stats().quotes, 0);
//! ```

use crate::Decimal;
use crate::backtest::data::MarketTick;
use crate::backtest::engine::{BacktestStrategy, SimulatedFill};
use crate::position::inventory::InventoryPosition;
use crate::strategy::avellaneda_stoikov::calculate_optimal_quotes;
use crate::strategy::quote::Quote;
use crate::strategy::walk_forward::{ConstantParameters, TimeWindow, WalkForwardReport};
use crate::types::error::{MMError, MMResult};

/// The parameters of the baseline's quotes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct QuotingParameters {
    /// Absolute price volatility per √year.
    pub volatility: Decimal,
    /// `k`, 1 / price units.
    pub order_intensity_k: Decimal,
}

/// One window of a schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ScheduleEntry {
    window: TimeWindow,
    /// `None` for a fold that could not calibrate.
    parameters: Option<QuotingParameters>,
}

/// Which parameters hold when: windows in order, with no overlap.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParameterSchedule {
    entries: Vec<ScheduleEntry>,
}

impl ParameterSchedule {
    /// Takes each fold's parameters for its test window. Under
    /// [`crate::strategy::walk_forward::UpdateRule::Frozen`] these are the first
    /// fold's.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] for no folds, test windows that
    /// overlap or go back, or a record whose training ends after its test window
    /// starts.
    pub fn from_report(report: &WalkForwardReport) -> MMResult<Self> {
        let mut entries = Vec::with_capacity(report.folds.len());
        for fold in &report.folds {
            if fold.record.training.end_ms > fold.split.test.start_ms {
                return Err(MMError::InvalidConfiguration(format!(
                    "training ends at {} after its test window starts at {}",
                    fold.record.training.end_ms, fold.split.test.start_ms
                )));
            }
            entries.push(ScheduleEntry {
                window: fold.split.test,
                parameters: fold.record.parameters().map(|p| QuotingParameters {
                    volatility: p.volatility,
                    order_intensity_k: p.order_intensity_k,
                }),
            });
        }
        Self::from_entries(entries)
    }

    /// Holds constant parameters for one window, to compare under the same rules.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] for an empty window, or a
    /// volatility or `k` that is not positive.
    pub fn constant(window: TimeWindow, constant: &ConstantParameters) -> MMResult<Self> {
        if constant.volatility <= Decimal::ZERO || constant.order_intensity_k <= Decimal::ZERO {
            return Err(MMError::InvalidConfiguration(format!(
                "constant parameters must be positive: {constant:?}"
            )));
        }
        Self::from_entries(vec![ScheduleEntry {
            window,
            parameters: Some(QuotingParameters {
                volatility: constant.volatility,
                order_intensity_k: constant.order_intensity_k,
            }),
        }])
    }

    fn from_entries(entries: Vec<ScheduleEntry>) -> MMResult<Self> {
        if entries.is_empty() {
            return Err(MMError::InvalidConfiguration(
                "a schedule needs a window".to_string(),
            ));
        }
        if let Some(bad) = entries
            .iter()
            .find(|e| e.window.start_ms >= e.window.end_ms)
        {
            return Err(MMError::InvalidConfiguration(format!(
                "empty window in a schedule: {:?}",
                bad.window
            )));
        }
        if let Some(bad) = entries
            .windows(2)
            .find(|pair| matches!(pair, [a, b] if b.window.start_ms < a.window.end_ms))
        {
            return Err(MMError::InvalidConfiguration(format!(
                "windows of a schedule overlap or go back: {bad:?}"
            )));
        }
        Ok(Self { entries })
    }

    /// The window holding `timestamp_ms`, if any.
    fn entry_at(&self, timestamp_ms: u64) -> Option<&ScheduleEntry> {
        let after = self
            .entries
            .partition_point(|e| e.window.start_ms <= timestamp_ms);
        after
            .checked_sub(1)
            .and_then(|index| self.entries.get(index))
            .filter(|e| timestamp_ms < e.window.end_ms)
    }

    /// The parameters holding at `timestamp_ms`. `None` outside every window or
    /// in a fold that could not calibrate.
    #[must_use]
    pub fn parameters_at(&self, timestamp_ms: u64) -> Option<QuotingParameters> {
        self.entry_at(timestamp_ms).and_then(|e| e.parameters)
    }
}

/// What a quoter did, tick by tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct QuoterStats {
    /// Ticks that got a quote.
    pub quotes: u64,
    /// Ticks outside every window of the schedule.
    pub outside_schedule: u64,
    /// Ticks in a fold that could not calibrate.
    pub without_parameters: u64,
    /// Ticks whose quote could not be computed.
    pub quote_errors: u64,
}

/// The Avellaneda-Stoikov baseline, quoting from a [`ParameterSchedule`].
#[derive(Debug, Clone)]
pub struct CalibratedQuoter {
    schedule: ParameterSchedule,
    risk_aversion: Decimal,
    horizon_ms: u64,
    order_size: Decimal,
    stats: QuoterStats,
}

impl CalibratedQuoter {
    /// Creates a quoter with a fixed time to terminal at each tick.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] for a risk aversion or an order
    /// size that is not positive.
    pub fn new(
        schedule: ParameterSchedule,
        risk_aversion: Decimal,
        horizon_ms: u64,
        order_size: Decimal,
    ) -> MMResult<Self> {
        if risk_aversion <= Decimal::ZERO || order_size <= Decimal::ZERO {
            return Err(MMError::InvalidConfiguration(format!(
                "risk aversion {risk_aversion} and order size {order_size} must be positive"
            )));
        }
        Ok(Self {
            schedule,
            risk_aversion,
            horizon_ms,
            order_size,
            stats: QuoterStats::default(),
        })
    }

    /// What the quoter did so far.
    #[must_use]
    pub fn stats(&self) -> QuoterStats {
        self.stats
    }

    /// The schedule it quotes from.
    #[must_use]
    pub fn schedule(&self) -> &ParameterSchedule {
        &self.schedule
    }
}

impl BacktestStrategy for CalibratedQuoter {
    fn on_tick(&mut self, tick: &MarketTick, position: &InventoryPosition) -> Option<Quote> {
        let Some(entry) = self.schedule.entry_at(tick.timestamp) else {
            self.stats.outside_schedule = self.stats.outside_schedule.saturating_add(1);
            return None;
        };
        let Some(parameters) = entry.parameters else {
            self.stats.without_parameters = self.stats.without_parameters.saturating_add(1);
            return None;
        };
        let Ok((bid_price, ask_price)) = calculate_optimal_quotes(
            tick.mid_price(),
            position.quantity,
            self.risk_aversion,
            parameters.volatility,
            self.horizon_ms,
            parameters.order_intensity_k,
        ) else {
            self.stats.quote_errors = self.stats.quote_errors.saturating_add(1);
            return None;
        };
        self.stats.quotes = self.stats.quotes.saturating_add(1);
        Some(Quote {
            bid_price,
            bid_size: self.order_size,
            ask_price,
            ask_size: self.order_size,
            timestamp: tick.timestamp,
        })
    }

    fn on_fill(&mut self, _fill: &SimulatedFill) {}

    fn cancels_when_silent(&self) -> bool {
        true
    }

    fn reset(&mut self) {
        self.stats = QuoterStats::default();
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
    #![allow(clippy::indexing_slicing, clippy::arithmetic_side_effects)]

    use super::*;
    use crate::dec;
    use crate::strategy::walk_forward::{
        CalibrationOutcome, CalibrationRecord, ChronologicalSplit, FoldReport, OutOfSampleError,
    };

    fn fold(train_end: u64, test_start: u64, test_end: u64) -> FoldReport {
        FoldReport {
            split: ChronologicalSplit {
                train: TimeWindow {
                    start_ms: 0,
                    end_ms: test_start,
                },
                test: TimeWindow {
                    start_ms: test_start,
                    end_ms: test_end,
                },
            },
            record: CalibrationRecord {
                training: TimeWindow {
                    start_ms: 0,
                    end_ms: train_end,
                },
                resolution_ms: 1,
                dataset_id: String::new(),
                dataset_hash: 0,
                price_samples: 0,
                fills: 0,
                exposure_ms: 0,
                open_ms: None,
                outcome: CalibrationOutcome::Insufficient(
                    crate::strategy::walk_forward::CalibrationFailure::FlatPrices,
                ),
            },
            calibrated: None,
            constant: OutOfSampleError {
                volatility: None,
                intensity: None,
                intensity_out_of_support: None,
            },
            k_change: None,
            volatility_change: None,
        }
    }

    #[test]
    fn test_from_report_refuses_training_that_ends_after_its_test_starts() {
        let report = WalkForwardReport {
            folds: vec![fold(101, 100, 200)],
        };
        assert!(matches!(
            ParameterSchedule::from_report(&report),
            Err(MMError::InvalidConfiguration(_))
        ));
        let report = WalkForwardReport {
            folds: vec![fold(100, 100, 200)],
        };
        assert!(ParameterSchedule::from_report(&report).is_ok());
    }

    #[test]
    fn test_from_report_refuses_no_folds_and_windows_that_overlap_or_go_back() {
        let of = |folds| ParameterSchedule::from_report(&WalkForwardReport { folds });
        assert!(of(vec![]).is_err());
        assert!(of(vec![fold(100, 100, 200), fold(100, 199, 300)]).is_err());
        assert!(of(vec![fold(100, 200, 300), fold(100, 100, 150)]).is_err());
        assert!(of(vec![fold(100, 100, 100)]).is_err());
        assert!(of(vec![fold(100, 100, 200), fold(100, 200, 300)]).is_ok());
    }

    #[test]
    fn test_a_fold_that_could_not_calibrate_holds_no_parameters() {
        let schedule = ParameterSchedule::from_report(&WalkForwardReport {
            folds: vec![fold(100, 100, 200)],
        })
        .unwrap();
        assert!(schedule.entry_at(150).is_some());
        assert_eq!(schedule.parameters_at(150), None);
    }

    fn constant() -> ConstantParameters {
        ConstantParameters {
            volatility: dec!(2),
            intensity_scale_a: dec!(1),
            order_intensity_k: dec!(1.5),
        }
    }

    #[test]
    fn test_constant_refuses_what_is_not_positive() {
        let window = TimeWindow {
            start_ms: 0,
            end_ms: 10,
        };
        let mut bad = constant();
        bad.volatility = Decimal::ZERO;
        assert!(ParameterSchedule::constant(window, &bad).is_err());
        let mut bad = constant();
        bad.order_intensity_k = dec!(-1);
        assert!(ParameterSchedule::constant(window, &bad).is_err());
        let empty = TimeWindow {
            start_ms: 10,
            end_ms: 10,
        };
        assert!(ParameterSchedule::constant(empty, &constant()).is_err());
        assert!(ParameterSchedule::constant(window, &constant()).is_ok());
    }

    #[test]
    fn test_windows_are_half_open() {
        let window = TimeWindow {
            start_ms: 100,
            end_ms: 200,
        };
        let schedule = ParameterSchedule::constant(window, &constant()).unwrap();
        assert!(schedule.parameters_at(99).is_none());
        assert!(schedule.parameters_at(100).is_some());
        assert!(schedule.parameters_at(199).is_some());
        assert!(schedule.parameters_at(200).is_none());
    }

    #[test]
    fn test_new_refuses_what_is_not_positive() {
        let window = TimeWindow {
            start_ms: 0,
            end_ms: 10,
        };
        let schedule = ParameterSchedule::constant(window, &constant()).unwrap();
        assert!(CalibratedQuoter::new(schedule.clone(), dec!(0), 1, dec!(1)).is_err());
        assert!(CalibratedQuoter::new(schedule.clone(), dec!(1), 1, dec!(0)).is_err());
        assert!(CalibratedQuoter::new(schedule, dec!(1), 1, dec!(1)).is_ok());
    }
}
