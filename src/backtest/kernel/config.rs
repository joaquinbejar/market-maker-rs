//! Kernel configuration, validation and the exactness bound `B`.

use crate::Decimal;
use crate::risk::RiskLimits;
use crate::types::error::{MMError, MMResult};

use super::encode::{CONFIG_MAGIC, FORMAT_VERSION, Writer, fnv1a128};
use super::grid::{Unit, pow10_u128};

/// Largest accepted `max_steps` (`2^60`), so `8 * S + 1` fits in `u64`.
pub const MAX_KERNEL_STEPS: u64 = 1 << 60;

/// Largest `Decimal` scale.
const MAX_SCALE: u32 = 28;

/// `2^96`: the `Decimal` mantissa bound.
const MANTISSA_LIMIT: u128 = 1 << 96;

/// Simulated latency of the kernel, in quanta of `quantum_ns`.
///
/// Every delay is `(base + jitter) * quantum_ns`, with the jitter drawn in
/// quanta from the kernel's latency stream exactly as the engine draws it
/// in milliseconds; with `quantum_ns = 1_000_000` the kernel reproduces an
/// engine run with the same delays in milliseconds.
///
/// As in the engine, messages due at a row's time run before the row is
/// matched, an order matches only on rows after the one it becomes active
/// on, and every quote cancels the resting orders first. So with rows every
/// `Δ`, a quote on every row, and `submit_q` and `cancel_q` both exactly `Δ`
/// with no jitter, no order ever fills: the cancel always lands on the first
/// row the order could fill on. See [`crate::backtest::latency`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelLatency {
    /// Length of one quantum, in nanoseconds; at least 1.
    pub quantum_ns: u64,
    /// Observation delay, in quanta.
    pub observation_q: u64,
    /// Submission delay, in quanta.
    pub submit_q: u64,
    /// Cancel delay, in quanta.
    pub cancel_q: u64,
    /// Acknowledgement delay, in quanta.
    pub ack_q: u64,
    /// Maximum jitter added to each delay, in quanta.
    pub jitter_q: u64,
}

impl Default for KernelLatency {
    fn default() -> Self {
        Self::zero(1_000_000)
    }
}

impl KernelLatency {
    /// Zero latency with the given quantum.
    #[must_use]
    pub fn zero(quantum_ns: u64) -> Self {
        Self {
            quantum_ns,
            observation_q: 0,
            submit_q: 0,
            cancel_q: 0,
            ack_q: 0,
            jitter_q: 0,
        }
    }

    /// Sets the four delays, in quanta.
    #[must_use]
    pub fn with_delays(
        mut self,
        observation_q: u64,
        submit_q: u64,
        cancel_q: u64,
        ack_q: u64,
    ) -> Self {
        self.observation_q = observation_q;
        self.submit_q = submit_q;
        self.cancel_q = cancel_q;
        self.ack_q = ack_q;
        self
    }

    /// Sets the maximum jitter, in quanta.
    #[must_use]
    pub fn with_jitter(mut self, jitter_q: u64) -> Self {
        self.jitter_q = jitter_q;
        self
    }

    /// `quanta * quantum_ns`, `None` on overflow.
    pub(super) fn ns(&self, quanta: u64) -> Option<u64> {
        quanta.checked_mul(self.quantum_ns)
    }

    /// Upper bound of the delay between the last row and any event it can
    /// cause: every delay at its maximum, plus jitter, summed.
    pub(super) fn max_total_ns(&self) -> Option<u64> {
        let quanta = self
            .observation_q
            .checked_add(self.submit_q)?
            .checked_add(self.cancel_q)?
            .checked_add(self.ack_q)?
            .checked_add(self.jitter_q.checked_mul(4)?)?;
        self.ns(quanta)
    }
}

/// Currency the kernel charges fees in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum KernelFee {
    /// Quote asset: `price * quantity * rate`, unrounded, as the engine's
    /// `FeeCurrency::Quote`.
    Quote,
    /// Base asset, rounded to `scale` places. Reserved for a later version:
    /// [`super::FinancialKernel::new`] rejects it.
    Base {
        /// Decimal places of the base asset.
        scale: u32,
    },
}

/// Fill model of the kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum KernelFillModel {
    /// The engine's `ImmediateFillModel`: a resting order fills completely
    /// at its limit price when the touch crosses it.
    Immediate,
    /// The engine's [`QueuePositionFillModel`]
    /// (`with_min_queue_time(rate, min_queue_ms)`): once the touch crosses a
    /// resting order, volume clears at `rate` per millisecond in queue, the
    /// touch size is ahead of an order at the touch and nothing is ahead of one
    /// the market trades through. What clears beyond that is the order's
    /// cumulative fill, so an order can fill over several rows
    /// ([`super::ExecKind::PartialFill`]).
    ///
    /// The rate is at least one lot per millisecond. When the rate times the
    /// time between rows exceeds every touch, a crossed order fills whole on
    /// the first row it can, exactly as [`KernelFillModel::Immediate`];
    /// [`KernelFillModel::QueuePositionRational`] clears less.
    ///
    /// [`QueuePositionFillModel`]: crate::backtest::QueuePositionFillModel
    QueuePosition {
        /// Volume cleared per millisecond in queue: positive, on the lot grid
        /// and at its normalised scale.
        rate: Decimal,
        /// Milliseconds in queue before any fill.
        min_queue_ms: u64,
    },
    /// The queue position model at a rate of `lots` lots every `per_ms`
    /// milliseconds, which can be below one lot per millisecond. After `t`
    /// milliseconds in queue the cleared volume is `floor(lots * t / per_ms)`
    /// lots, computed from the order's entry to the queue, so no remainder is
    /// lost between rows; the rest is as [`KernelFillModel::QueuePosition`].
    ///
    /// It equals the engine's [`QueuePositionFillModel`] at a rate of
    /// `lots * lot_size / per_ms` per millisecond whenever that cleared
    /// volume is a whole number of lots, and with `per_ms = 1` it equals
    /// [`KernelFillModel::QueuePosition`] at `lots * lot_size`. Otherwise the
    /// engine can fill a fraction of a lot, and the kernel fills the whole
    /// lots below it.
    ///
    /// [`QueuePositionFillModel`]: crate::backtest::QueuePositionFillModel
    QueuePositionRational {
        /// Lots cleared every `per_ms` milliseconds in queue; at least 1.
        lots: u64,
        /// Milliseconds in which `lots` clear; at least 1.
        per_ms: u64,
        /// Milliseconds in queue before any fill.
        min_queue_ms: u64,
    },
}

/// The validated fill model, in lots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ValidFill {
    /// [`KernelFillModel::Immediate`].
    Immediate,
    /// [`KernelFillModel::QueuePosition`] and
    /// [`KernelFillModel::QueuePositionRational`].
    QueuePosition {
        /// Lots cleared every `per_ms` milliseconds.
        rate_lots: u64,
        /// Milliseconds in which `rate_lots` clear; 1 for
        /// [`KernelFillModel::QueuePosition`].
        per_ms: u64,
        /// Milliseconds in queue before any fill.
        min_queue_ms: u64,
    },
}

/// A balance gate on the kernel's own ledger, as `margin_rs::risk::RiskGate`
/// decides for a spot instrument in `backtest::MarginAccount`
/// (`margin` feature): fully collateralised, with no leverage.
///
/// - A buy reserves its notional at its limit price plus its largest fee,
///   `max(fee_rate, 0)` of that notional, from the quote balance:
///   [`KernelConfig::initial_capital`] plus the fill cash flows, minus the
///   fees and the reservations of the live buys.
/// - A sell reserves its quantity from the base balance: `base_deposit`
///   plus the venue position, minus the live sells. Selling short draws on
///   the deposit.
///
/// An order beyond the balance is reduced to what the balance covers, as the
/// gate reduces it, or refused when that is less than one unit of the lot's
/// grid; a reduction that is not a whole number of lots is refused, as the
/// engine refuses it. A live order is one submitted and not yet filled,
/// cancelled at the venue or rejected there: a requested cancel keeps its
/// reservation, since the order can still fill. A fill releases what it
/// filled. The gate runs after [`KernelConfig::limits`], on the lots they
/// approved, and its decisions are [`super::GateRecord`]s with
/// [`super::GateCheck::Margin`].
///
/// The kernel and an engine with a `MarginAccount` decide alike when the
/// account's quote asset has the kernel's money scale (tick, lot and fee
/// scales summed), its base asset and the instrument's quantity the lot's
/// scale, the instrument's price scale the tick's, the deposits are
/// `initial_capital` and `base_deposit`, `max_fee_rate` is `max(fee_rate,
/// 0)` with fees in quote, and no other limit of the account is set.
/// Margin requirements of a derivative, maintenance and liquidation are not
/// modelled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct KernelMargin {
    /// Base deposit, on the lot grid; zero or more. Sells draw on it.
    pub base_deposit: Decimal,
}

impl KernelMargin {
    /// A balance gate with `base_deposit` of base; the quote balance is the
    /// kernel's initial capital.
    #[must_use]
    pub fn new(base_deposit: Decimal) -> Self {
        Self { base_deposit }
    }
}

/// The validated balance gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ValidMargin {
    /// Base deposit, in lots.
    pub(super) base_lots: u64,
    /// The fee rate mantissa reserved on buys: `max(rate_num, 0)`.
    pub(super) fee_num: i128,
}

/// User facing kernel configuration, in `Decimal`.
///
/// [`super::FinancialKernel::new`] validates it into integers and checks
/// the bounds that keep every ledger value exact (see the module docs).
#[derive(Debug, Clone, PartialEq)]
pub struct KernelConfig {
    /// Price grid step; positive.
    pub tick_size: Decimal,
    /// Quantity grid step; positive.
    pub lot_size: Decimal,
    /// Initial capital; an exact multiple of the money unit.
    pub initial_capital: Decimal,
    /// Fee rate applied to the notional; any sign.
    pub fee_rate: Decimal,
    /// Fee currency.
    pub fee_currency: KernelFee,
    /// Largest price, on the tick grid.
    pub max_price: Decimal,
    /// Largest order quantity, on the lot grid.
    pub max_order_size: Decimal,
    /// Largest number of decisions, at most [`MAX_KERNEL_STEPS`].
    pub max_steps: u64,
    /// Simulated latency.
    pub latency: KernelLatency,
    /// Optional position and notional limits, checked as the engine's
    /// `LimitsPolicy` checks them (no circuit breaker).
    pub limits: Option<RiskLimits>,
    /// Fill model.
    pub fill_model: KernelFillModel,
    /// Optional balance gate on the kernel's ledger ([`KernelMargin`]).
    pub margin: Option<KernelMargin>,
}

impl KernelConfig {
    /// A configuration with zero capital, zero fees, zero latency and no
    /// limits.
    #[must_use]
    pub fn new(
        tick_size: Decimal,
        lot_size: Decimal,
        max_price: Decimal,
        max_order_size: Decimal,
        max_steps: u64,
    ) -> Self {
        Self {
            tick_size,
            lot_size,
            initial_capital: Decimal::ZERO,
            fee_rate: Decimal::ZERO,
            fee_currency: KernelFee::Quote,
            max_price,
            max_order_size,
            max_steps,
            latency: KernelLatency::default(),
            limits: None,
            fill_model: KernelFillModel::Immediate,
            margin: None,
        }
    }

    /// Sets the initial capital.
    #[must_use]
    pub fn with_initial_capital(mut self, capital: Decimal) -> Self {
        self.initial_capital = capital;
        self
    }

    /// Sets the fee rate.
    #[must_use]
    pub fn with_fee_rate(mut self, rate: Decimal) -> Self {
        self.fee_rate = rate;
        self
    }

    /// Sets the latency.
    #[must_use]
    pub fn with_latency(mut self, latency: KernelLatency) -> Self {
        self.latency = latency;
        self
    }

    /// Sets the fill model.
    #[must_use]
    pub fn with_fill_model(mut self, fill_model: KernelFillModel) -> Self {
        self.fill_model = fill_model;
        self
    }

    /// Sets the limits.
    #[must_use]
    pub fn with_limits(mut self, limits: RiskLimits) -> Self {
        self.limits = Some(limits);
        self
    }

    /// Sets the balance gate.
    #[must_use]
    pub fn with_margin(mut self, margin: KernelMargin) -> Self {
        self.margin = Some(margin);
        self
    }

    /// Canonical encoding (`MMKC`), the input of the fingerprint.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut w = Writer::new();
        w.bytes(&CONFIG_MAGIC);
        w.u16(FORMAT_VERSION);
        w.decimal(self.tick_size);
        w.decimal(self.lot_size);
        w.decimal(self.initial_capital);
        w.decimal(self.fee_rate);
        match self.fee_currency {
            KernelFee::Quote => w.u16(1),
            KernelFee::Base { scale } => {
                w.u16(2);
                w.u32(scale);
            }
        }
        w.decimal(self.max_price);
        w.decimal(self.max_order_size);
        w.u64(self.max_steps);
        let l = self.latency;
        for v in [
            l.quantum_ns,
            l.observation_q,
            l.submit_q,
            l.cancel_q,
            l.ack_q,
            l.jitter_q,
        ] {
            w.u64(v);
        }
        match &self.limits {
            None => w.u8(0),
            Some(limits) => {
                w.u8(1);
                w.decimal(limits.max_position);
                w.decimal(limits.max_notional);
                w.decimal(limits.scaling_factor);
            }
        }
        match self.fill_model {
            KernelFillModel::Immediate => w.u16(1),
            KernelFillModel::QueuePosition { rate, min_queue_ms } => {
                w.u16(2);
                w.decimal(rate);
                w.u64(min_queue_ms);
            }
            KernelFillModel::QueuePositionRational {
                lots,
                per_ms,
                min_queue_ms,
            } => {
                w.u16(3);
                w.u64(lots);
                w.u64(per_ms);
                w.u64(min_queue_ms);
            }
        }
        // A section that a configuration without the balance gate does not
        // write, so its bytes and fingerprint stay as they were.
        if let Some(margin) = &self.margin {
            w.bytes(b"MG");
            w.decimal(margin.base_deposit);
        }
        w.finish()
    }

    /// FNV-1a-128 of [`KernelConfig::encode`].
    #[must_use]
    pub fn fingerprint(&self) -> u128 {
        fnv1a128(&self.encode())
    }
}

/// Validated configuration: integer grid, bounds and limits.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct ValidatedConfig {
    pub(super) tick: Unit,
    pub(super) lot: Unit,
    /// Fee rate mantissa and scale.
    pub(super) rate_num: i128,
    pub(super) fs: u32,
    /// Money scale `s = ts + ls + fs`.
    pub(super) s: u32,
    /// Initial capital in U.
    pub(super) c0_u: i128,
    /// Largest price in ticks.
    pub(super) p: u64,
    /// Smallest admissible price in ticks (1; 0 only in test kernels).
    pub(super) min_price: u64,
    /// Largest order size in lots.
    pub(super) q: u64,
    pub(super) s_max: u64,
    pub(super) x_eff: u64,
    pub(super) bound_b: u128,
    pub(super) latency: KernelLatency,
    pub(super) limits: Option<RiskLimits>,
    pub(super) margin: Option<ValidMargin>,
    pub(super) fill: ValidFill,
    /// [`CapacityNeed::partial_reports_per_order`] on the dataset.
    pub(super) partial_reports_per_order: u64,
    pub(super) fingerprint: u128,
}

impl ValidatedConfig {
    /// `F = 10^fs`.
    pub(super) fn fee_scale(&self) -> Option<i128> {
        i128::try_from(pow10_u128(self.fs)?).ok()
    }

    /// Notional of `price` ticks times `qty` lots, in U.
    pub(super) fn notional_u(&self, price: u64, qty: u64) -> Option<i128> {
        i128::from(price)
            .checked_mul(i128::from(self.tick.mantissa))?
            .checked_mul(i128::from(qty))?
            .checked_mul(i128::from(self.lot.mantissa))?
            .checked_mul(self.fee_scale()?)
    }

    /// Quote fee of `price` ticks times `qty` lots, in U.
    pub(super) fn fee_u(&self, price: u64, qty: u64) -> Option<i128> {
        i128::from(price)
            .checked_mul(i128::from(self.tick.mantissa))?
            .checked_mul(i128::from(qty))?
            .checked_mul(i128::from(self.lot.mantissa))?
            .checked_mul(self.rate_num)
    }

    /// Value of `pos` lots at `mark2` half ticks, in half-U.
    pub(super) fn position_value_h(&self, pos: i64, mark2: u64) -> Option<i128> {
        i128::from(pos)
            .checked_mul(i128::from(mark2))?
            .checked_mul(i128::from(self.tick.mantissa))?
            .checked_mul(i128::from(self.lot.mantissa))?
            .checked_mul(self.fee_scale()?)
    }

    /// Largest magnitude accepted for a half-U value (`2B`).
    pub(super) fn half_bound(&self) -> u128 {
        self.bound_b.saturating_mul(2)
    }

    /// Largest drawdown, in half-U: `peak - equity <= 2B` U, so `4B`.
    pub(super) fn drawdown_bound(&self) -> u128 {
        self.bound_b.saturating_mul(4)
    }
}

/// Splits a fee rate into mantissa and scale.
fn rate_parts(rate: Decimal) -> (i128, u32) {
    let normal = rate.normalize();
    if normal.is_zero() {
        (0, 0)
    } else {
        (normal.mantissa(), normal.scale())
    }
}

fn config_error(message: String) -> MMError {
    MMError::InvalidConfiguration(message)
}

/// Options for the test-only unchecked validation.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct Relax {
    /// Skip the exactness bound `B` and the width checks.
    pub(super) bounds: bool,
    /// Admit price 0 (to reach the venue validation reject).
    pub(super) zero_price: bool,
}

/// Validates `cfg` into integers and checks the bounds of the module docs.
pub(super) fn validate(cfg: &KernelConfig, relax: Relax) -> MMResult<ValidatedConfig> {
    let tick = Unit::new(cfg.tick_size).ok_or_else(|| {
        config_error(format!("tick_size must be positive, got {}", cfg.tick_size))
    })?;
    let lot = Unit::new(cfg.lot_size)
        .ok_or_else(|| config_error(format!("lot_size must be positive, got {}", cfg.lot_size)))?;
    // The engine computes with the raw `Decimal` scales: trailing zeros
    // would raise its mantissas above the ones the bound `B` is computed
    // for, so they are rejected rather than normalised.
    for (name, value) in [
        ("tick_size", cfg.tick_size),
        ("lot_size", cfg.lot_size),
        ("fee_rate", cfg.fee_rate),
    ] {
        if value.scale() != value.normalize().scale() {
            return Err(config_error(format!(
                "{name} {value} has trailing zeros; write it at its normalised scale ({})",
                value.normalize()
            )));
        }
    }
    match cfg.fee_currency {
        KernelFee::Quote => {}
        KernelFee::Base { scale } => {
            return Err(config_error(format!(
                "base fee currency (scale {scale}) is not supported by kernel v1"
            )));
        }
    }
    let fill = match cfg.fill_model {
        KernelFillModel::Immediate => ValidFill::Immediate,
        KernelFillModel::QueuePosition { rate, min_queue_ms } => {
            // As the other grid values: the engine computes with raw scales.
            if rate.scale() != rate.normalize().scale() {
                return Err(config_error(format!(
                    "fill rate {rate} has trailing zeros; write it at its normalised scale ({})",
                    rate.normalize()
                )));
            }
            let rate_lots = lot
                .count(rate)
                .and_then(|lots| u64::try_from(lots).ok())
                .filter(|lots| *lots > 0)
                .ok_or_else(|| {
                    config_error(format!(
                        "fill rate must be a positive number of lots of {}, got {rate}",
                        cfg.lot_size
                    ))
                })?;
            ValidFill::QueuePosition {
                rate_lots,
                per_ms: 1,
                min_queue_ms,
            }
        }
        KernelFillModel::QueuePositionRational {
            lots,
            per_ms,
            min_queue_ms,
        } => {
            if lots == 0 || per_ms == 0 {
                return Err(config_error(format!(
                    "fill rate must be at least one lot in at least one ms, got {lots} lots \
                     per {per_ms} ms"
                )));
            }
            ValidFill::QueuePosition {
                rate_lots: lots,
                per_ms,
                min_queue_ms,
            }
        }
    };
    if cfg.latency.quantum_ns == 0 {
        return Err(config_error(
            "latency quantum_ns must be at least 1, got 0".to_string(),
        ));
    }
    if cfg.max_steps > MAX_KERNEL_STEPS {
        return Err(config_error(format!(
            "max_steps must be at most {MAX_KERNEL_STEPS}, got {}",
            cfg.max_steps
        )));
    }
    let (rate_num, fs) = rate_parts(cfg.fee_rate);
    let s = tick
        .scale
        .checked_add(lot.scale)
        .and_then(|v| v.checked_add(fs))
        .ok_or_else(|| config_error("money scale overflows".to_string()))?;
    if !relax.bounds && s.saturating_add(1) > MAX_SCALE {
        return Err(config_error(format!(
            "money scale s + 1 = {} exceeds {MAX_SCALE} (tick scale {}, lot scale {}, fee scale {fs})",
            s.saturating_add(1),
            tick.scale,
            lot.scale
        )));
    }
    let money = Unit {
        mantissa: 1,
        scale: s,
        value: Decimal::try_from_i128_with_scale(1, s.min(MAX_SCALE))
            .map_err(|e| config_error(format!("money unit: {e}")))?,
    };
    if cfg.initial_capital.scale() > s {
        return Err(config_error(format!(
            "initial_capital {} has scale {} above the money scale {s}",
            cfg.initial_capital,
            cfg.initial_capital.scale()
        )));
    }
    let c0_u = money.count(cfg.initial_capital).ok_or_else(|| {
        config_error(format!(
            "initial_capital {} is not a whole number of money units 1e-{s}",
            cfg.initial_capital
        ))
    })?;
    let p = tick
        .count(cfg.max_price)
        .and_then(|v| u64::try_from(v).ok())
        .filter(|v| *v >= 1)
        .ok_or_else(|| {
            config_error(format!(
                "max_price {} must be a positive multiple of tick_size {} fitting u64 ticks",
                cfg.max_price, cfg.tick_size
            ))
        })?;
    let q = lot
        .count(cfg.max_order_size)
        .and_then(|v| u64::try_from(v).ok())
        .filter(|v| *v >= 1)
        .ok_or_else(|| {
            config_error(format!(
                "max_order_size {} must be a positive multiple of lot_size {} fitting u64 lots",
                cfg.max_order_size, cfg.lot_size
            ))
        })?;
    if let Some(limits) = &cfg.limits
        && (limits.max_position <= Decimal::ZERO || limits.max_notional <= Decimal::ZERO)
    {
        return Err(config_error(format!(
            "limits must be positive, got max_position {} and max_notional {}",
            limits.max_position, limits.max_notional
        )));
    }
    let margin = match &cfg.margin {
        None => None,
        Some(m) => {
            let base_lots = if m.base_deposit.is_zero() {
                Some(0)
            } else {
                lot.count(m.base_deposit)
                    .and_then(|v| u64::try_from(v).ok())
                    .filter(|v| i64::try_from(*v).is_ok())
            };
            let base_lots = base_lots.ok_or_else(|| {
                config_error(format!(
                    "margin base_deposit {} must be zero or a positive multiple of lot_size {} \
                     fitting i64 lots",
                    m.base_deposit, cfg.lot_size
                ))
            })?;
            Some(ValidMargin {
                base_lots,
                fee_num: rate_num.max(0),
            })
        }
    };
    let (x_eff, bound_b) = if relax.bounds {
        (i64::MAX.unsigned_abs(), u128::MAX / 40)
    } else {
        bound(cfg, &tick, &lot, rate_num, fs, c0_u, p, q)?
    };
    let fingerprint = cfg.fingerprint();
    Ok(ValidatedConfig {
        tick,
        lot,
        rate_num,
        fs,
        s,
        c0_u,
        p,
        min_price: if relax.zero_price { 0 } else { 1 },
        q,
        s_max: cfg.max_steps,
        x_eff,
        bound_b,
        latency: cfg.latency,
        limits: cfg.limits.clone(),
        margin,
        fill,
        partial_reports_per_order: 0,
        fingerprint,
    })
}

/// Position bound `X_eff` and exactness bound `B` (module docs, bounds).
#[allow(clippy::too_many_arguments)] // Every input is one symbol of the bound formula.
fn bound(
    cfg: &KernelConfig,
    tick: &Unit,
    lot: &Unit,
    rate_num: i128,
    fs: u32,
    c0_u: i128,
    p: u64,
    q: u64,
) -> MMResult<(u64, u128)> {
    let overflow = |what: &str| config_error(format!("bound {what} overflows u128"));
    if p.checked_mul(2).is_none() {
        return Err(config_error(format!(
            "2 * max_price ticks ({p}) must fit u64 for the mid"
        )));
    }
    let f = pow10_u128(fs).ok_or_else(|| overflow("F"))?;
    let r = rate_num.unsigned_abs();
    let pmt = u128::from(p)
        .checked_mul(u128::from(tick.mantissa))
        .ok_or_else(|| overflow("P*mt"))?;
    let qml = u128::from(q)
        .checked_mul(u128::from(lot.mantissa))
        .ok_or_else(|| overflow("Q*ml"))?;
    let n = pmt
        .checked_mul(qml)
        .and_then(|v| v.checked_mul(f))
        .ok_or_else(|| overflow("N"))?;
    let fee = pmt
        .checked_mul(qml)
        .and_then(|v| v.checked_mul(r))
        .ok_or_else(|| overflow("Fee"))?;
    let two_s = u128::from(cfg.max_steps)
        .checked_mul(2)
        .ok_or_else(|| overflow("2S"))?;
    let x_cap = two_s
        .checked_mul(u128::from(q))
        .ok_or_else(|| overflow("2S*Q"))?;
    let i64_max = i64::MAX.unsigned_abs();
    if x_cap > u128::from(i64_max) {
        return Err(config_error(format!(
            "2 * max_steps * max_order_size lots = {x_cap} does not fit i64 lots"
        )));
    }
    let x_limit = cfg.limits.as_ref().and_then(|limits| {
        let lots = limits.max_position.checked_div(lot.value)?.floor();
        u64::try_from(lots.mantissa())
            .ok()
            .filter(|_| lots.scale() == 0)
    });
    let x_eff = match x_limit {
        Some(x) => u128::from(x).min(x_cap),
        None => x_cap,
    };
    let b = Some(c0_u.unsigned_abs())
        .and_then(|c0| {
            let fills = two_s.checked_mul(n.checked_add(fee)?)?;
            let held = x_eff
                .checked_mul(pmt)?
                .checked_mul(u128::from(lot.mantissa))?
                .checked_mul(f)?;
            c0.checked_add(fills)?.checked_add(held)
        })
        .ok_or_else(|| overflow("B"))?;
    let factor: u128 = if fs == 0 { 20 } else { 2 };
    let scaled = b
        .checked_mul(factor)
        .ok_or_else(|| overflow("factor * B"))?;
    if scaled >= MANTISSA_LIMIT {
        return Err(config_error(format!(
            "exactness bound {factor} * B = {scaled} must be below 2^96 (B = {b})"
        )));
    }
    let i128_cap = i128::MAX.unsigned_abs() / 20;
    if b > i128_cap {
        return Err(config_error(format!(
            "bound B = {b} exceeds i128::MAX / 20"
        )));
    }
    let x_eff = u64::try_from(x_eff).map_err(|_| overflow("X_eff"))?;
    Ok((x_eff, b))
}

/// Capacities the kernel needs for one configuration and dataset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapacityNeed {
    /// Most observations in flight at once; `OBS` must be at least this.
    pub obs: u64,
    /// Most decisions inside one closing window (`D`).
    pub decisions_in_window: u64,
    /// Smallest `ORD`: `2 * (D + 1) + 2`. `MSG` must be at least `2 * ORD`
    /// and `DLV` at least `(3 + partial_reports_per_order) * ORD`.
    pub ord: u64,
    /// Most partial fill reports of one order in flight at once (`R`): the
    /// most rows within the acknowledgement delay and its jitter, since an
    /// order reports at most once per row. Zero for the immediate model.
    pub partial_reports_per_order: u64,
}

/// Computes the [`CapacityNeed`] from the timestamps and the latency.
pub(super) fn capacity_need(
    timestamps: &[u64],
    latency: &KernelLatency,
    fill_model: &KernelFillModel,
) -> MMResult<CapacityNeed> {
    let overflow = || config_error("capacity window overflows".to_string());
    let obs_window = latency
        .ns(latency
            .observation_q
            .checked_add(latency.jitter_q)
            .ok_or_else(overflow)?)
        .ok_or_else(overflow)?;
    let close_quanta = latency
        .submit_q
        .checked_add(latency.cancel_q)
        .and_then(|v| v.checked_add(latency.ack_q))
        .and_then(|v| v.checked_add(latency.jitter_q.checked_mul(6)?))
        .ok_or_else(overflow)?;
    let window = latency.ns(close_quanta).ok_or_else(overflow)?;
    let mut obs: u64 = 0;
    let mut d: u64 = 0;
    let mut lo = 0usize;
    let mut hi = 0usize;
    for (j, ts) in timestamps.iter().enumerate() {
        // Rows still in flight when row `j` is pushed: `ts_i + W > ts_j`.
        while let Some(first) = timestamps.get(lo) {
            // Two `u64` values summed in a `u128` cannot overflow.
            #[allow(clippy::arithmetic_side_effects)]
            let ends = u128::from(*first) + u128::from(obs_window);
            if ends > u128::from(*ts) {
                break;
            }
            lo = lo.saturating_add(1);
        }
        let in_flight =
            u64::try_from(j.saturating_sub(lo).saturating_add(1)).map_err(|_| overflow())?;
        obs = obs.max(in_flight);
        // Rows with `ts_j <= ts_i <= ts_j + window`.
        if hi < j {
            hi = j;
        }
        while let Some(next) = timestamps.get(hi) {
            // Two `u64` values summed in a `u128` cannot overflow.
            #[allow(clippy::arithmetic_side_effects)]
            let horizon = u128::from(*ts) + u128::from(window);
            if u128::from(*next) > horizon {
                break;
            }
            hi = hi.saturating_add(1);
        }
        let in_window = u64::try_from(hi.saturating_sub(j)).map_err(|_| overflow())?;
        d = d.max(in_window);
    }
    let ord = d
        .checked_add(1)
        .and_then(|v| v.checked_mul(2))
        .and_then(|v| v.checked_add(2))
        .ok_or_else(overflow)?;
    let partial_reports_per_order = match fill_model {
        KernelFillModel::Immediate => 0,
        KernelFillModel::QueuePosition { .. } | KernelFillModel::QueuePositionRational { .. } => {
            rows_within(
                timestamps,
                latency
                    .ns(latency
                        .ack_q
                        .checked_add(latency.jitter_q)
                        .ok_or_else(overflow)?)
                    .ok_or_else(overflow)?,
            )?
        }
    };
    Ok(CapacityNeed {
        obs,
        decisions_in_window: d,
        ord,
        partial_reports_per_order,
    })
}

/// The most rows `i <= j` with `ts_i + window >= ts_j`, at least one per row:
/// a report of row `i` is delivered by `ts_i + window`, so these are the rows
/// whose reports can still be in flight when row `j` is matched.
fn rows_within(timestamps: &[u64], window: u64) -> MMResult<u64> {
    let mut most = 0u64;
    let mut lo = 0usize;
    for (j, ts) in timestamps.iter().enumerate() {
        while let Some(first) = timestamps.get(lo) {
            if first.saturating_add(window) >= *ts {
                break;
            }
            lo = lo.saturating_add(1);
        }
        let rows = u64::try_from(j.saturating_sub(lo).saturating_add(1))
            .map_err(|_| config_error("the number of rows does not fit u64".to_string()))?;
        most = most.max(rows);
    }
    Ok(most)
}
