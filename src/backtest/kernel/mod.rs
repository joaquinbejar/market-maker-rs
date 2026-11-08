//! The financial environment kernel (feature `kernel`).
//!
//! [`FinancialKernel`] re-implements the backtest engine's tick loop as a
//! resumable state machine over a fixed width, `Copy` [`KernelState`]. It is
//! meant to be driven step by step by an external RL trainer through a bridge
//! crate: [`FinancialKernel::reset`] builds the state at the first decision
//! point, [`FinancialKernel::step`] applies one [`KernelAction`] and advances
//! to the next decision point, [`FinancialKernel::observe`] and
//! [`FinancialKernel::mask`] give what the agent may see. A copy of the state
//! is a snapshot; [`encode`] and [`decode`] give its canonical bytes, and
//! `decode` accepts only states that satisfy every invariant `step` relies
//! on (it does not prove a state was reached by stepping).
//!
//! # Contract
//!
//! Inside the supported subset the kernel reproduces a
//! [`BacktestEngine`](crate::backtest::BacktestEngine) run exactly, report
//! for report, fill for fill, and ledger value for ledger value, with no
//! tolerances:
//!
//! - single linear spot instrument on a tick and lot grid;
//! - the immediate or the queue position fill model, on the lot grid, no
//!   slippage, quote currency fees (any sign);
//! - no pretrade policy, the engine's `LimitsPolicy` (no circuit breaker),
//!   or, with the balance gate ([`KernelMargin`]), a
//!   `backtest::MarginAccount` set up as
//!   [`KernelMargin`] states (`margin` feature);
//! - the built-in bookkeeping (no external accountant);
//! - all four latencies and jitter, in quanta of `quantum_ns`
//!   (`quantum_ns = 1_000_000` matches the engine's milliseconds);
//! - datasets with non-decreasing timestamps (the kernel rejects others at
//!   load; the engine clamps them);
//! - actions `Hold`, `Quote` and `QuoteError` (an engine `None`); the
//!   one-sided `OneSided` and the cancel-only `Cancel` actions have no
//!   engine counterpart and are outside the subset.
//!
//! # Ordering
//!
//! Rows are processed in file order. Within one row: venue messages,
//! deliveries and observations due by the row time run first, in
//! `(time, class)` order with class venue < delivery < observation; then
//! the resting orders are matched in book (activation) order; then the row's observation
//! is queued and the events due by the row time run again; finally the venue
//! ledger is marked at the row's mid. A decision point is the moment an
//! observation is popped. Every `step` call while running consumes one of
//! `max_steps`; at `max_steps` the run is `Truncated`. At the end of the
//! data the queued reports are delivered and pending venue messages and
//! observations are dropped and counted. Calling `step` after the run ended
//! changes nothing.
//!
//! # Agent visibility
//!
//! [`Observation`] and [`ActionMask`] depend only on delivered reports, the
//! agent's own actions, the last observed row, the step count, the status
//! and the configuration. Delivered reports carry two fields beyond the
//! engine's report, [`ReportRec::origin`] and [`ReportRec::cancel_pending`],
//! which let the agent close an order exactly when its last report arrives.
//! Admission reserves one order slot, two venue messages and three
//! deliveries per open order, so the internal queues can never overflow:
//! partial fill reports come on top of those three, and
//! [`FinancialKernel::new`] requires room for them
//! ([`CapacityNeed::partial_reports_per_order`]).
//!
//! # Exactness bounds
//!
//! [`FinancialKernel::new`] computes, with `tick = mt·10^-ts`,
//! `lot = ml·10^-ls`, `rate = r·10^-fs` and the money unit `U = 10^-s`,
//! `s = ts + ls + fs`:
//!
//! - `N = P·mt·Q·ml·10^fs` and `Fee = P·mt·Q·ml·|r|` (per order, whose fills
//!   sum to at most `Q` lots, in U),
//! - `X_eff = min(max_position / lot, 2·S·Q)`,
//! - `B = C0 + 2S·(N + Fee) + X_eff·P·mt·ml·10^fs`,
//!
//! and requires `s + 1 <= 28`, `2B < 2^96` (`20B < 2^96` when `fs = 0`),
//! `2P` within `u64`, `2S·Q` within `i64`, and no time overflow. With the
//! queue position model it also requires the volume cleared by the last row,
//! the touch sizes and the largest order to fit a `Decimal` mantissa at the
//! lot scale, so the engine's model never overflows or rounds them. Inside
//! these bounds every `Decimal` ledger operation is exact, so the engine's
//! saturating arithmetic never saturates and the integer shadow equals it.
//!
//! # A hedge lane
//!
//! [`HedgedKernel`] adds a second instrument to hedge in, outside the
//! supported subset. Without it, nothing here changes.
//!
//! # Example
//!
//! ```rust
//! use market_maker_rs::backtest::MarketTick;
//! use market_maker_rs::backtest::kernel::{FinancialKernel, KernelAction, KernelConfig, KernelDataset};
//! use market_maker_rs::dec;
//!
//! let cfg = KernelConfig::new(dec!(0.01), dec!(1), dec!(1000), dec!(10), 100)
//!     .with_initial_capital(dec!(10000));
//! let ticks: Vec<MarketTick> = (0..5)
//!     .map(|i| MarketTick::new(1_000 + i, dec!(100.00), dec!(1), dec!(100.02), dec!(1)))
//!     .collect();
//! let data = KernelDataset::from_ticks(&ticks, &cfg)?;
//! let kernel: FinancialKernel<'_> = FinancialKernel::new(&cfg, &data)?;
//! let mut state = kernel.reset(7);
//! while state.status().is_running() {
//!     let obs = kernel.observe(&state);
//!     let action = KernelAction::Quote { bid: obs.bid, ask: obs.ask, bid_lots: 1, ask_lots: 1 };
//!     kernel.step(&mut state, action);
//! }
//! assert_eq!(state.cursor(), 5);
//! # Ok::<(), market_maker_rs::types::error::MMError>(())
//! ```

mod account;
mod adapter;
mod book;
mod config;
mod data;
mod encode;
mod gate;
mod grid;
mod hedge;
mod ledger;
mod observe;
mod rng;
mod state;
mod step;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_expiry;

pub use adapter::{
    AvellanedaQuoter, ERR_KERNEL_MISMATCH, ERR_PRICE_OUT_OF_RANGE, ERR_SIZE_OUT_OF_RANGE,
    GlftCounts, GlftForm, GlftQuoteForm, GlftQuoter, QuoteAdapter, QuoteSource, SourceQuote,
    quote_error_code, quote_error_name,
};
pub use book::{ExecKind, FillRec, Origin, ReportRec};
pub use config::{
    CapacityNeed, KernelConfig, KernelFee, KernelFillModel, KernelLatency, KernelMargin,
    MAX_KERNEL_STEPS,
};
pub use data::{DIGEST_FNV1A128, DatasetId, KernelDataset, KernelTick};
pub use encode::{
    CONFIG_MAGIC, FORMAT_VERSION, STATE_MAGIC, decode, encode, fingerprint, fnv1a128,
};
pub use gate::{GateCheck, GateVerdict};
pub use hedge::{
    CombinedLimits, FundingRate, HEDGE_FORMAT_VERSION, HEDGED_STATE_MAGIC, HedgeCounters,
    HedgeEvent, HedgeFill, HedgeInstrument, HedgeLane, HedgeLaneConfig, HedgeRefusal, HedgeSummary,
    HedgeView, HedgedAction, HedgedKernel, HedgedOutcome, HedgedState,
};
pub use observe::{ActionGrid, ActionMask, Observation};
pub use rng::{KernelRng, RNG_ALGO_XORSHIFT64STAR};
pub use state::{
    ActionStatus, Counters, ErrCode, EventSink, GateRecord, HaltReason, KernelAction, KernelEvent,
    KernelState, NoSink, Phase, QueueLengths, QuoteSide, RejectReason, Status, StepOutcome,
    TermReason,
};

use crate::Decimal;
use crate::types::action::Dir;
use crate::types::error::{MMError, MMResult};

use account::held;
use book::Arr;
use config::{Relax, ValidFill, ValidatedConfig};
use gate::GateState;
use grid::{half_units_decimal, scaled_decimal};
use ledger::{AgentView, VenueLedger};
use step::Em;

/// The kernel: a validated configuration over a borrowed dataset.
///
/// `ORD`, `MSG`, `DLV` and `OBS` are the capacities of the order slots,
/// venue messages, deliveries and observations; [`FinancialKernel::new`]
/// checks they are large enough for the dataset and latency
/// ([`capacity_need`]).
#[derive(Debug, Clone)]
pub struct FinancialKernel<
    'd,
    const ORD: usize = 16,
    const MSG: usize = 64,
    const DLV: usize = 128,
    const OBS: usize = 64,
> {
    cfg: ValidatedConfig,
    data: &'d KernelDataset,
}

/// Venue side totals in `Decimal`, as the engine's `get_state` and
/// `BacktestResult` report them.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct VenueSummary {
    /// Position quantity.
    pub position: Decimal,
    /// Average entry price.
    pub avg_entry: Decimal,
    /// Realized cash flow.
    pub realized: Decimal,
    /// Unrealized value at the last mark.
    pub unrealized: Decimal,
    /// `realized + unrealized`.
    pub total_pnl: Decimal,
    /// Fees paid.
    pub total_fees: Decimal,
    /// `total_pnl - total_fees`.
    pub net_pnl: Decimal,
    /// Largest absolute position.
    pub max_position: Decimal,
    /// Largest drawdown.
    pub max_drawdown: Decimal,
    /// Peak equity.
    pub peak_equity: Decimal,
}

impl<'d, const ORD: usize, const MSG: usize, const DLV: usize, const OBS: usize>
    FinancialKernel<'d, ORD, MSG, DLV, OBS>
{
    /// Validates `cfg` against the bounds and builds a kernel over `data`.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] if the configuration is
    /// invalid or breaks a bound (module docs), if `tick_size`, `lot_size`
    /// or `fee_rate` carry trailing zeros or `initial_capital` a scale above
    /// the money scale (the engine computes with raw scales, so these are
    /// rejected rather than normalised), if `data` was built on
    /// another grid, if a capacity or the length does not fit `u32`, or if
    /// a capacity is below [`capacity_need`].
    pub fn new(cfg: &KernelConfig, data: &'d KernelDataset) -> MMResult<Self> {
        Self::build(cfg, data, Relax::default(), false)
    }

    fn build(
        cfg: &KernelConfig,
        data: &'d KernelDataset,
        relax: Relax,
        skip_capacity: bool,
    ) -> MMResult<Self> {
        let mut validated = config::validate(cfg, relax)?;
        let grid = data.grid;
        if grid.tick_m != validated.tick.mantissa
            || grid.tick_s != validated.tick.scale
            || grid.lot_m != validated.lot.mantissa
            || grid.lot_s != validated.lot.scale
            || grid.max_price != validated.p
        {
            return Err(MMError::InvalidConfiguration(
                "dataset was built on another tick, lot or max_price grid".to_string(),
            ));
        }
        let tail = validated.latency.max_total_ns().ok_or_else(|| {
            MMError::InvalidConfiguration("total latency overflows u64".to_string())
        })?;
        if data.id.last_ts.checked_add(tail).is_none() {
            return Err(MMError::InvalidConfiguration(format!(
                "last timestamp {} ns plus latency {tail} ns overflows u64",
                data.id.last_ts
            )));
        }
        for (name, value) in [
            ("ORD", ORD),
            ("MSG", MSG),
            ("DLV", DLV),
            ("OBS", OBS),
            ("len", data.len()),
        ] {
            if u32::try_from(value).is_err() {
                return Err(MMError::InvalidConfiguration(format!(
                    "{name} = {value} does not fit u32"
                )));
            }
        }
        check_fill_model(&validated, data)?;
        if matches!(validated.fill, ValidFill::QueuePosition { .. }) || !skip_capacity {
            let need =
                config::capacity_need(&timestamps(data), &validated.latency, &cfg.fill_model)?;
            validated.partial_reports_per_order = need.partial_reports_per_order;
            if !skip_capacity {
                check_capacity::<ORD, MSG, DLV, OBS>(&need)?;
            }
        }
        Ok(Self {
            cfg: validated,
            data,
        })
    }

    /// A kernel without the bound and capacity checks, for tests that
    /// reach the halts and capacity rejects.
    #[cfg(test)]
    pub(crate) fn new_relaxed(
        cfg: &KernelConfig,
        data: &'d KernelDataset,
        bounds: bool,
        zero_price: bool,
        skip_capacity: bool,
    ) -> MMResult<Self> {
        Self::build(cfg, data, Relax { bounds, zero_price }, skip_capacity)
    }

    /// The dataset.
    #[must_use]
    pub fn data(&self) -> &'d KernelDataset {
        self.data
    }

    /// Fingerprint of the configuration.
    #[must_use]
    pub fn config_fingerprint(&self) -> u128 {
        self.cfg.fingerprint
    }

    /// Exactness bound `B`, in money units.
    #[must_use]
    pub fn bound(&self) -> u128 {
        self.cfg.bound_b
    }

    /// Money scale `s` (one money unit is `10^-s`).
    #[must_use]
    pub fn money_scale(&self) -> u32 {
        self.cfg.s
    }

    /// `value` money units as a `Decimal`.
    #[must_use]
    pub fn money(&self, value: i128) -> Option<Decimal> {
        scaled_decimal(value, self.cfg.s)
    }

    /// `half` half money units as a `Decimal`.
    #[must_use]
    pub fn half_money(&self, half: i128) -> Option<Decimal> {
        half_units_decimal(half, self.cfg.s)
    }

    /// `ticks` as a price.
    #[must_use]
    pub fn price(&self, ticks: u64) -> Option<Decimal> {
        self.cfg.tick.to_decimal(i128::from(ticks))
    }

    /// `lots` as a quantity.
    #[must_use]
    pub fn quantity(&self, lots: u64) -> Option<Decimal> {
        self.cfg.lot.to_decimal(i128::from(lots))
    }

    /// The state at the first decision point (or the end of the run), with
    /// the latency stream seeded by `seed`.
    #[must_use]
    pub fn reset(&self, seed: u64) -> KernelState<ORD, MSG, DLV, OBS> {
        self.reset_with(seed, &mut NoSink)
    }

    /// As [`FinancialKernel::reset`], sending the events to `sink`.
    pub fn reset_with<E: EventSink>(
        &self,
        seed: u64,
        sink: &mut E,
    ) -> KernelState<ORD, MSG, DLV, OBS> {
        let ledger = VenueLedger::new(self.cfg.c0_u);
        let mut s = KernelState {
            data_id: self.data.id,
            cfg_fp: self.cfg.fingerprint,
            format_version: FORMAT_VERSION,
            phase: Phase::ObsQueued,
            resume: Phase::ObsQueued,
            cursor: 0,
            cur_row: None,
            now_ns: 0,
            step: 0,
            status: Status::Running,
            rng_latency: KernelRng::new(seed),
            rng_fill: KernelRng::new(seed),
            stamp_seq: 0,
            next_msg_seq: 0,
            next_order_id: 1,
            next_exec_id: 1,
            last_delivery_ns: 0,
            last_observed_ns: 0,
            orders: Arr::EMPTY,
            msgs: Arr::EMPTY,
            awaiting: Arr::EMPTY,
            venue: ledger.unwrap_or_else(empty_ledger),
            gate: GateState::NEW,
            deliveries: Arr::EMPTY,
            observed: Arr::EMPTY,
            agent: AgentView::NEW,
            account: Arr::EMPTY,
            last: None,
            counters: Counters::ZERO,
        };
        let mut em = Em::new(sink);
        if ledger.is_none() {
            step::halt(&mut s, HaltReason::LedgerOverflow, &mut em);
            return s;
        }
        step::advance(&self.cfg, self.data, &mut s, &mut em);
        s
    }

    /// Applies `action` and advances to the next decision point.
    pub fn step(
        &self,
        s: &mut KernelState<ORD, MSG, DLV, OBS>,
        action: KernelAction,
    ) -> StepOutcome {
        self.step_with(s, action, &mut NoSink)
    }

    /// As [`FinancialKernel::step`], sending the events to `sink`.
    pub fn step_with<E: EventSink>(
        &self,
        s: &mut KernelState<ORD, MSG, DLV, OBS>,
        action: KernelAction,
        sink: &mut E,
    ) -> StepOutcome {
        self.step_combined(s, action, None, sink)
    }

    /// As [`FinancialKernel::step_with`], with the combined limits of a
    /// hedged kernel binding the orders it submits.
    fn step_combined<E: EventSink>(
        &self,
        s: &mut KernelState<ORD, MSG, DLV, OBS>,
        action: KernelAction,
        mut combined: step::Combined<'_>,
        sink: &mut E,
    ) -> StepOutcome {
        let mut em = Em::new(sink);
        let terminal = ActionStatus::Rejected(RejectReason::Terminal);
        if !self.owns(s) {
            if s.status.is_running() {
                step::halt(s, HaltReason::StateMismatch, &mut em);
            }
            return StepOutcome {
                applied: terminal,
                status: s.status,
                events: em.count,
            };
        }
        if !s.status.is_running() || s.phase != Phase::AwaitDecision {
            return StepOutcome {
                applied: terminal,
                status: s.status,
                events: 0,
            };
        }
        let counted = s
            .step
            .checked_add(1)
            .and_then(|step| Some((step, s.counters.actions.checked_add(1)?)));
        let Some((step, actions)) = counted else {
            step::halt(s, HaltReason::CounterOverflow, &mut em);
            return StepOutcome {
                applied: terminal,
                status: s.status,
                events: em.count,
            };
        };
        s.step = step;
        s.counters.actions = actions;
        let applied = match action {
            KernelAction::Hold => ActionStatus::Applied,
            KernelAction::QuoteError(code) => {
                // At most one per step, and steps are at most S <= 2^60.
                let Some(errors) = s.counters.quote_errors.checked_add(1) else {
                    step::halt(s, HaltReason::CounterOverflow, &mut em);
                    return StepOutcome {
                        applied: terminal,
                        status: s.status,
                        events: em.count,
                    };
                };
                s.counters.quote_errors = errors;
                ActionStatus::QuoteError(code)
            }
            KernelAction::Quote {
                bid,
                ask,
                bid_lots,
                ask_lots,
            } => self.admit_apply(
                s,
                step::admit(&self.cfg, s, bid, ask, bid_lots, ask_lots),
                step::Legs {
                    legs: [
                        Some((Dir::Buy, bid, bid_lots)),
                        Some((Dir::Sell, ask, ask_lots)),
                    ],
                    crossed: bid >= ask,
                    life: None,
                },
                combined
                    .as_mut()
                    .map(|(check, rejects)| (*check, &mut **rejects)),
                &mut em,
            ),
            KernelAction::ExpiringQuote {
                bid,
                ask,
                bid_lots,
                ask_lots,
                life_ns,
            } => self.admit_apply(
                s,
                step::admit(&self.cfg, s, bid, ask, bid_lots, ask_lots),
                step::Legs {
                    legs: [
                        Some((Dir::Buy, bid, bid_lots)),
                        Some((Dir::Sell, ask, ask_lots)),
                    ],
                    crossed: bid >= ask,
                    life: Some(life_ns),
                },
                combined
                    .as_mut()
                    .map(|(check, rejects)| (*check, &mut **rejects)),
                &mut em,
            ),
            KernelAction::OneSided { side, price, lots } => {
                let dir = match side {
                    QuoteSide::Bid => Dir::Buy,
                    QuoteSide::Ask => Dir::Sell,
                };
                self.admit_apply(
                    s,
                    step::admit(&self.cfg, s, price, price, lots, lots),
                    step::Legs {
                        legs: [Some((dir, price, lots)), None],
                        crossed: false,
                        life: None,
                    },
                    combined
                        .as_mut()
                        .map(|(check, rejects)| (*check, &mut **rejects)),
                    &mut em,
                )
            }
            KernelAction::Cancel => self.admit_apply(
                s,
                Ok(()),
                step::Legs {
                    legs: [None, None],
                    crossed: false,
                    life: None,
                },
                combined
                    .as_mut()
                    .map(|(check, rejects)| (*check, &mut **rejects)),
                &mut em,
            ),
        };
        if !s.status.is_running() {
            return StepOutcome {
                applied,
                status: s.status,
                events: em.count,
            };
        }
        em.event(KernelEvent::Decided(applied));
        if let Err(reason) = step::record(s, applied) {
            step::halt(s, reason, &mut em);
        } else {
            s.phase = s.resume;
            step::advance(&self.cfg, self.data, s, &mut em);
        }
        if em.overflow && s.status.is_running() {
            step::halt(s, HaltReason::CounterOverflow, &mut em);
        }
        StepOutcome {
            applied,
            status: s.status,
            events: em.count,
        }
    }

    /// Admission then the transaction; a transaction failure rolls back,
    /// halts and reports `Rejected(Terminal)`.
    fn admit_apply<E: EventSink>(
        &self,
        s: &mut KernelState<ORD, MSG, DLV, OBS>,
        admitted: Result<(), RejectReason>,
        legs: step::Legs,
        combined: step::Combined<'_>,
        em: &mut Em<'_, E>,
    ) -> ActionStatus {
        match admitted {
            Err(reason) => ActionStatus::Rejected(reason),
            Ok(()) => match step::apply_quote(&self.cfg, s, legs, combined, em) {
                Ok(()) => ActionStatus::Applied,
                // The transaction was rolled back and the run halted: the
                // action never took effect. `StepOutcome::status` carries
                // the halt reason.
                Err(reason) => {
                    step::halt(s, reason, em);
                    ActionStatus::Rejected(RejectReason::Terminal)
                }
            },
        }
    }

    /// Returns true if `s` belongs to this kernel's dataset and
    /// configuration.
    #[must_use]
    pub fn owns(&self, s: &KernelState<ORD, MSG, DLV, OBS>) -> bool {
        s.data_id == self.data.id && s.cfg_fp == self.cfg.fingerprint
    }

    /// The agent's observation.
    ///
    /// A state from another kernel gives an empty observation with status
    /// `Halted(StateMismatch)`: its rows and ledger use another grid.
    #[must_use]
    pub fn observe(&self, s: &KernelState<ORD, MSG, DLV, OBS>) -> Observation {
        if !self.owns(s) {
            return Observation {
                row: None,
                ts_ns: 0,
                bid: 0,
                bid_lots: 0,
                ask: 0,
                ask_lots: 0,
                position: Decimal::ZERO,
                avg_entry: Decimal::ZERO,
                delivered_fills: 0,
                delivered_local_rejects: 0,
                open_orders: 0,
                awaiting_cancel_outcome: 0,
                step: 0,
                status: Status::Halted(HaltReason::StateMismatch),
                rejected: [0; 8],
            };
        }
        let row = s
            .agent
            .last_obs_row
            .and_then(|r| self.data.row(r).map(|t| (r, *t)));
        let awaiting = s
            .account
            .as_slice()
            .iter()
            .filter(|o| o.awaiting_cancel_outcome)
            .count();
        Observation {
            row: row.map(|(r, _)| r),
            ts_ns: row.map_or(0, |(_, t)| t.ts_ns),
            bid: row.map_or(0, |(_, t)| t.bid),
            bid_lots: row.map_or(0, |(_, t)| t.bid_lots),
            ask: row.map_or(0, |(_, t)| t.ask),
            ask_lots: row.map_or(0, |(_, t)| t.ask_lots),
            position: s.agent.pos.quantity,
            avg_entry: s.agent.pos.avg_entry,
            delivered_fills: s.agent.delivered_fills,
            delivered_local_rejects: s.agent.delivered_local_rejects,
            open_orders: held(&s.account),
            awaiting_cancel_outcome: u64::try_from(awaiting).unwrap_or(u64::MAX),
            step: s.step,
            status: s.status,
            rejected: s.counters.rejected,
        }
    }

    /// Admission mask over `grid` around the observed mid; equal to the
    /// admission predicate of [`FinancialKernel::step`] by construction.
    #[must_use]
    pub fn mask<const K: usize, const M: usize>(
        &self,
        s: &KernelState<ORD, MSG, DLV, OBS>,
        grid: &ActionGrid<K, M>,
    ) -> ActionMask {
        let identity = self.owns(s);
        if !identity {
            return ActionMask {
                global_quote_ok: false,
                bid: 0,
                ask: 0,
                size: 0,
                refusal: Some(RejectReason::Terminal),
            };
        }
        let deciding = s.status.is_running() && s.phase == Phase::AwaitDecision;
        let capacity = step::capacity_ok::<ORD, MSG, DLV>(held(&s.account));
        let refusal = if deciding {
            capacity.err()
        } else {
            Some(RejectReason::Terminal)
        };
        let mid2 = s
            .agent
            .last_obs_row
            .and_then(|r| self.data.row(r))
            .and_then(|t| t.mid2());
        let price_ok =
            |p: Option<u64>| p.is_some_and(|p| p >= self.cfg.min_price && p <= self.cfg.p);
        let mut bid = 0u64;
        let mut ask = 0u64;
        let mut size = 0u64;
        if let Some(mid2) = mid2 {
            for i in 0..K.min(64) {
                let bit = 1u64
                    .checked_shl(u32::try_from(i).unwrap_or(64))
                    .unwrap_or(0);
                if price_ok(grid.bid_price(mid2, i)) {
                    bid |= bit;
                }
                if price_ok(grid.ask_price(mid2, i)) {
                    ask |= bit;
                }
            }
        }
        for (i, lots) in grid.sizes().iter().enumerate().take(64) {
            let bit = 1u64
                .checked_shl(u32::try_from(i).unwrap_or(64))
                .unwrap_or(0);
            if *lots >= 1 && *lots <= self.cfg.q {
                size |= bit;
            }
        }
        ActionMask {
            global_quote_ok: refusal.is_none(),
            bid,
            ask,
            size,
            refusal,
        }
    }

    /// The quote at grid indices `(bid, ask, bid size, ask size)` around the
    /// observed mid, if the prices are representable (`None` for a state
    /// from another kernel).
    #[must_use]
    pub fn grid_action<const K: usize, const M: usize>(
        &self,
        s: &KernelState<ORD, MSG, DLV, OBS>,
        grid: &ActionGrid<K, M>,
        index: (usize, usize, usize, usize),
    ) -> Option<KernelAction> {
        if !self.owns(s) {
            return None;
        }
        let mid2 = s
            .agent
            .last_obs_row
            .and_then(|r| self.data.row(r))
            .and_then(|t| t.mid2())?;
        Some(KernelAction::Quote {
            bid: grid.bid_price(mid2, index.0)?,
            ask: grid.ask_price(mid2, index.1)?,
            bid_lots: *grid.sizes().get(index.2)?,
            ask_lots: *grid.sizes().get(index.3)?,
        })
    }

    /// Venue side totals in `Decimal`.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidMarketState`] for a state from another
    /// kernel (its ledger uses another money scale), and
    /// [`MMError::NumericalError`] if a value does not convert (unreachable
    /// inside the bounds).
    pub fn venue_summary(&self, s: &KernelState<ORD, MSG, DLV, OBS>) -> MMResult<VenueSummary> {
        if !self.owns(s) {
            return Err(MMError::InvalidMarketState(
                "venue summary of a state from another dataset or configuration".to_string(),
            ));
        }
        let fail = || MMError::NumericalError("venue summary does not convert".to_string());
        let v = &s.venue;
        let unrealized_h = match v.last_mark2 {
            Some(mark2) => self
                .cfg
                .position_value_h(s.gate.venue_pos_lots, mark2)
                .ok_or_else(fail)?,
            None => 0,
        };
        let unrealized = self.half_money(unrealized_h).ok_or_else(fail)?;
        let total_pnl = v.realized.checked_add(unrealized).ok_or_else(fail)?;
        let total_fees = self.money(v.fixed_fees_u).ok_or_else(fail)?;
        Ok(VenueSummary {
            position: v.pos.quantity,
            avg_entry: v.pos.avg_entry,
            realized: v.realized,
            unrealized,
            total_pnl,
            total_fees,
            net_pnl: total_pnl.checked_sub(total_fees).ok_or_else(fail)?,
            max_position: self.quantity(v.max_abs_pos).ok_or_else(fail)?,
            max_drawdown: self.half_money(v.max_dd_h).ok_or_else(fail)?,
            peak_equity: self.half_money(v.peak_h).ok_or_else(fail)?,
        })
    }
}

/// Capacities needed by `cfg` over `data`; choose `ORD`, `MSG`, `DLV` and
/// `OBS` at least this large.
///
/// # Errors
///
/// Returns [`MMError::InvalidConfiguration`] if a window overflows.
pub fn capacity_need(cfg: &KernelConfig, data: &KernelDataset) -> MMResult<CapacityNeed> {
    config::capacity_need(&timestamps(data), &cfg.latency, &cfg.fill_model)
}

/// The queue position model on `data` stays inside the integers the engine's
/// `Decimal` model computes with: below `2^96` at the lot scale, its mantissa
/// never overflows or rounds, so its overflow branch cannot be reached.
fn check_fill_model(cfg: &ValidatedConfig, data: &KernelDataset) -> MMResult<()> {
    const LIMIT: u128 = 1 << 96;
    let ValidFill::QueuePosition {
        rate_lots, per_ms, ..
    } = cfg.fill
    else {
        return Ok(());
    };
    let unit = u128::from(cfg.lot.mantissa);
    let fits = |lots: u128| lots.checked_mul(unit).is_some_and(|m| m < LIMIT);
    let last_ms = data.id.last_ts.checked_div(data::NS_PER_MS).unwrap_or(0);
    let cleared = u128::from(rate_lots)
        .checked_mul(u128::from(last_ms))
        .and_then(|v| v.checked_div(u128::from(per_ms)));
    if !cleared.is_some_and(fits) {
        return Err(MMError::InvalidConfiguration(format!(
            "fill rate {rate_lots} lots per {per_ms} ms up to {last_ms} ms is not exact in a \
             Decimal"
        )));
    }
    let largest = data
        .ticks
        .iter()
        .map(|t| t.bid_lots.max(t.ask_lots))
        .fold(cfg.q, u64::max);
    if !fits(u128::from(largest)) {
        return Err(MMError::InvalidConfiguration(format!(
            "size of {largest} lots is not exact in a Decimal for the queue position model"
        )));
    }
    Ok(())
}

fn empty_ledger() -> VenueLedger {
    VenueLedger {
        cash_u: 0,
        fixed_fees_u: 0,
        base_fee_units: 0,
        pos: ledger::InvDec::FLAT,
        realized: Decimal::ZERO,
        last_mark2: None,
        peak_h: 0,
        max_dd_h: 0,
        max_abs_pos: 0,
    }
}

fn timestamps(data: &KernelDataset) -> Vec<u64> {
    data.ticks.iter().map(|t| t.ts_ns).collect()
}

fn check_capacity<const ORD: usize, const MSG: usize, const DLV: usize, const OBS: usize>(
    need: &CapacityNeed,
) -> MMResult<()> {
    let as_u64 = |v: usize| u64::try_from(v).unwrap_or(u64::MAX);
    let (ord, msg, dlv, obs) = (as_u64(ORD), as_u64(MSG), as_u64(DLV), as_u64(OBS));
    let fail = |name: &str, have: u64, want: u64| {
        Err(MMError::InvalidConfiguration(format!(
            "capacity {name} = {have} is below the required {want}"
        )))
    };
    if obs < need.obs {
        return fail("OBS", obs, need.obs);
    }
    if ord < need.ord {
        return fail("ORD", ord, need.ord);
    }
    let msg_need = ord.saturating_mul(2);
    if msg < msg_need {
        return fail("MSG", msg, msg_need);
    }
    let dlv_need = ord.saturating_mul(need.partial_reports_per_order.saturating_add(3));
    if dlv < dlv_need {
        return fail("DLV", dlv, dlv_need);
    }
    Ok(())
}
