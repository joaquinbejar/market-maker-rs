//! Canonical encoding v1 of the kernel state, `decode` with full
//! validation, and FNV-1a-128.
//!
//! Little endian, fixed width, no `usize`. Booleans are `u8` 0 or 1, enums
//! use fixed codes that are never reused, an option is a `u8` tag (0 or 1)
//! followed by the value, a `Decimal` is `(i128 mantissa, u8 scale)`, and a
//! collection is `capacity: u32, len: u32` followed by the occupied entries
//! in canonical order.
//!
//! # Field order (state, magic `MMKS`)
//!
//! | # | Field | Encoding |
//! |---|---|---|
//! | 1 | magic, format, latency and fill stream algorithm tags | `[u8; 4]`, `u16`, `u8`, `u8` |
//! | 2 | capacities `ORD`, `MSG`, `DLV`, `OBS` | `u32` each |
//! | 3 | dataset id: scheme, len, first ts, last ts, digest, external | `u16`, `u64`, `u64`, `u64`, `u128`, `[u8; 32]` |
//! | 4 | configuration fingerprint | `u128` |
//! | 5 | phase, resume phase, cursor, current row, now, step, status | `u8`, `u8`, `u64`, option `u64`, `u64`, `u64`, (`u8`, `u16`) |
//! | 6 | latency stream, fill stream: seed, state | `u64`, `u64` each |
//! | 7 | stamp seq, next msg seq, next order id, next exec id, last delivery, last observation | `u64` each |
//! | 8 | resting orders, in book (activation) order | collection of orders |
//! | 9 | venue messages, by `(eff, seq)` | collection of messages |
//! | 10 | orders awaiting a cancel outcome, by id | collection of orders |
//! | 11 | venue ledger | see `write_ledger` |
//! | 12 | gate: venue position, live buy, live sell, mark | `i64`, `u64`, `u64`, option `u64` |
//! | 13 | deliveries, FIFO order | collection of deliveries |
//! | 14 | observations, FIFO order | collection of `(ready, row)` |
//! | 15 | agent view: position, last observed row, delivered local rejects, delivered fills | |
//! | 16 | account, by id | collection of `(id, awaiting outcome)` |
//! | 17 | last outcome, counters | |

use crate::Decimal;
use crate::types::action::Dir;
use crate::types::error::{MMError, MMResult};

use core::num::NonZeroU64;
use std::collections::BTreeMap;

use super::FinancialKernel;
use super::account::OpenOrder;
use super::book::{
    Arr, Blank, Delivery, ExecKind, FillRec, MsgKind, ObsEntry, OrderState, Origin, ReportRec,
    VenueMsg, VenueOrder, dir_code, dir_from_code,
};
use super::config::{ValidFill, ValidatedConfig};
use super::data::DatasetId;
use super::gate::GateState;
use super::grid::scaled_decimal;
use super::ledger::{AgentView, InvDec, VenueLedger};
use super::observe::write_status;
use super::rng::{KernelRng, RNG_ALGO_XORSHIFT64STAR};
use super::state::{
    ActionStatus, Counters, HaltReason, KernelState, Phase, RejectReason, Status, TermReason,
};

/// Magic of an encoded state.
pub const STATE_MAGIC: [u8; 4] = *b"MMKS";
/// Magic of an encoded configuration.
pub const CONFIG_MAGIC: [u8; 4] = *b"MMKC";
/// Encoding format version.
pub const FORMAT_VERSION: u16 = 1;

const FNV_OFFSET: u128 = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d;
const FNV_PRIME: u128 = 0x0000_0000_0100_0000_0000_0000_0000_013B;

/// Incremental FNV-1a-128.
#[derive(Debug, Clone, Copy)]
pub(super) struct Fnv128(u128);

impl Fnv128 {
    pub(super) fn new() -> Self {
        Self(FNV_OFFSET)
    }

    pub(super) fn update(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 ^= u128::from(*byte);
            // The hash is defined with a wrapping multiply.
            self.0 = self.0.wrapping_mul(FNV_PRIME);
        }
    }

    pub(super) fn finish(self) -> u128 {
        self.0
    }
}

/// FNV-1a-128 of `bytes`.
#[must_use]
pub fn fnv1a128(bytes: &[u8]) -> u128 {
    let mut hash = Fnv128::new();
    hash.update(bytes);
    hash.finish()
}

/// Little endian writer.
#[derive(Debug, Default)]
pub(super) struct Writer(Vec<u8>);

impl Writer {
    pub(super) fn new() -> Self {
        Self(Vec::new())
    }
    pub(super) fn bytes(&mut self, b: &[u8]) {
        self.0.extend_from_slice(b);
    }
    pub(super) fn u8(&mut self, v: u8) {
        self.0.push(v);
    }
    pub(super) fn bool(&mut self, v: bool) {
        self.u8(u8::from(v));
    }
    pub(super) fn u16(&mut self, v: u16) {
        self.bytes(&v.to_le_bytes());
    }
    pub(super) fn u32(&mut self, v: u32) {
        self.bytes(&v.to_le_bytes());
    }
    pub(super) fn u64(&mut self, v: u64) {
        self.bytes(&v.to_le_bytes());
    }
    pub(super) fn i64(&mut self, v: i64) {
        self.bytes(&v.to_le_bytes());
    }
    pub(super) fn u128(&mut self, v: u128) {
        self.bytes(&v.to_le_bytes());
    }
    pub(super) fn i128(&mut self, v: i128) {
        self.bytes(&v.to_le_bytes());
    }
    pub(super) fn option_u64(&mut self, v: Option<u64>) {
        match v {
            None => self.u8(0),
            Some(v) => {
                self.u8(1);
                self.u64(v);
            }
        }
    }
    pub(super) fn decimal(&mut self, d: Decimal) {
        self.i128(d.mantissa());
        // A Decimal scale is at most 28.
        self.u8(u8::try_from(d.scale()).unwrap_or(u8::MAX));
    }
    fn len(&mut self, n: usize) {
        self.u32(u32::try_from(n).unwrap_or(u32::MAX));
    }
    pub(super) fn finish(self) -> Vec<u8> {
        self.0
    }
}

fn err(message: impl Into<String>) -> MMError {
    MMError::InvalidMarketState(format!("kernel state decode: {}", message.into()))
}

/// Little endian reader that never panics.
pub(super) struct Reader<'a> {
    pub(super) bytes: &'a [u8],
    pub(super) pos: usize,
}

impl<'a> Reader<'a> {
    pub(super) fn take(&mut self, n: usize) -> MMResult<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| err("length overflow"))?;
        let slice = self
            .bytes
            .get(self.pos..end)
            .ok_or_else(|| err(format!("truncated at byte {}", self.pos)))?;
        self.pos = end;
        Ok(slice)
    }
    pub(super) fn arr<const N: usize>(&mut self) -> MMResult<[u8; N]> {
        <[u8; N]>::try_from(self.take(N)?).map_err(|_| err("short read"))
    }
    pub(super) fn u8(&mut self) -> MMResult<u8> {
        Ok(u8::from_le_bytes(self.arr()?))
    }
    pub(super) fn bool(&mut self) -> MMResult<bool> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            v => Err(err(format!("invalid bool {v}"))),
        }
    }
    pub(super) fn u16(&mut self) -> MMResult<u16> {
        Ok(u16::from_le_bytes(self.arr()?))
    }
    pub(super) fn u32(&mut self) -> MMResult<u32> {
        Ok(u32::from_le_bytes(self.arr()?))
    }
    pub(super) fn u64(&mut self) -> MMResult<u64> {
        Ok(u64::from_le_bytes(self.arr()?))
    }
    pub(super) fn i64(&mut self) -> MMResult<i64> {
        Ok(i64::from_le_bytes(self.arr()?))
    }
    pub(super) fn u128(&mut self) -> MMResult<u128> {
        Ok(u128::from_le_bytes(self.arr()?))
    }
    pub(super) fn i128(&mut self) -> MMResult<i128> {
        Ok(i128::from_le_bytes(self.arr()?))
    }
    pub(super) fn option_u64(&mut self) -> MMResult<Option<u64>> {
        Ok(if self.bool()? {
            Some(self.u64()?)
        } else {
            None
        })
    }
    pub(super) fn decimal(&mut self) -> MMResult<Decimal> {
        let mantissa = self.i128()?;
        let scale = self.u8()?;
        if scale > 28 || mantissa.unsigned_abs() >= (1u128 << 96) {
            return Err(err(format!(
                "invalid decimal (mantissa {mantissa}, scale {scale})"
            )));
        }
        Decimal::try_from_i128_with_scale(mantissa, u32::from(scale))
            .map_err(|e| err(e.to_string()))
    }
    pub(super) fn dir(&mut self) -> MMResult<Dir> {
        let code = self.u8()?;
        dir_from_code(code).ok_or_else(|| err(format!("invalid side {code}")))
    }
    fn collection<T: Blank, const N: usize>(
        &mut self,
        what: &str,
        mut item: impl FnMut(&mut Self) -> MMResult<T>,
    ) -> MMResult<Arr<T, N>> {
        let cap = self.u32()?;
        if usize::try_from(cap).ok() != Some(N) {
            return Err(err(format!("{what} capacity {cap} differs from {N}")));
        }
        let len = self.u32()?;
        let mut out = Arr::<T, N>::EMPTY;
        for _ in 0..len {
            let value = item(self)?;
            out.push(value)
                .map_err(|_| err(format!("{what} length {len} exceeds capacity {N}")))?;
        }
        Ok(out)
    }
}

fn write_order(w: &mut Writer, o: &VenueOrder) {
    w.u64(o.id);
    w.u8(dir_code(o.side));
    w.u64(o.price);
    w.u64(o.qty);
    w.u64(o.filled);
    w.u8(o.state.code());
    w.u64(o.active_after_ns);
    w.bool(o.cancel_inflight);
}

fn read_order(r: &mut Reader<'_>) -> MMResult<VenueOrder> {
    let id = r.u64()?;
    let side = r.dir()?;
    let price = r.u64()?;
    let qty = r.u64()?;
    let filled = r.u64()?;
    let code = r.u8()?;
    let state =
        OrderState::from_code(code).ok_or_else(|| err(format!("invalid order state {code}")))?;
    Ok(VenueOrder {
        id,
        side,
        price,
        qty,
        filled,
        state,
        active_after_ns: r.u64()?,
        cancel_inflight: r.bool()?,
    })
}

fn action_dir(side: crate::execution::Action) -> u8 {
    Dir::try_from(side).map_or(0, dir_code)
}

fn write_report(w: &mut Writer, rep: &ReportRec) {
    w.u64(rep.ts_ns);
    w.u64(rep.seq);
    w.u64(rep.order_id);
    w.u8(action_dir(rep.side));
    w.u8(rep.kind.code());
    w.u8(rep.origin.code());
    w.bool(rep.cancel_pending);
    for v in [rep.price, rep.qty, rep.last, rep.cum] {
        w.u64(v);
    }
    w.i128(rep.fee_u);
}

fn read_report(r: &mut Reader<'_>) -> MMResult<ReportRec> {
    let ts_ns = r.u64()?;
    let seq = r.u64()?;
    let order_id = r.u64()?;
    let side = r.dir()?.into();
    let kind_code = r.u8()?;
    let kind = ExecKind::from_code(kind_code)
        .ok_or_else(|| err(format!("invalid report kind {kind_code}")))?;
    let origin_code = r.u8()?;
    let origin = Origin::from_code(origin_code)
        .ok_or_else(|| err(format!("invalid origin {origin_code}")))?;
    Ok(ReportRec {
        ts_ns,
        seq,
        order_id,
        side,
        kind,
        origin,
        cancel_pending: r.bool()?,
        price: r.u64()?,
        qty: r.u64()?,
        last: r.u64()?,
        cum: r.u64()?,
        fee_u: r.i128()?,
    })
}

fn write_fill(w: &mut Writer, f: &FillRec) {
    w.u64(f.ts_ns);
    w.u64(f.exec_id);
    w.u64(f.order_id);
    w.u8(action_dir(f.side));
    w.u64(f.price);
    w.u64(f.qty);
    w.i128(f.fee_u);
}

fn read_fill(r: &mut Reader<'_>) -> MMResult<FillRec> {
    Ok(FillRec {
        ts_ns: r.u64()?,
        exec_id: r.u64()?,
        order_id: r.u64()?,
        side: r.dir()?.into(),
        price: r.u64()?,
        qty: r.u64()?,
        fee_u: r.i128()?,
    })
}

fn write_inv(w: &mut Writer, p: &InvDec) {
    w.decimal(p.quantity);
    w.decimal(p.avg_entry);
    w.u64(p.last_update);
}

fn read_inv(r: &mut Reader<'_>) -> MMResult<InvDec> {
    Ok(InvDec {
        quantity: r.decimal()?,
        avg_entry: r.decimal()?,
        last_update: r.u64()?,
    })
}

fn write_ledger(w: &mut Writer, v: &VenueLedger) {
    w.i128(v.cash_u);
    w.i128(v.fixed_fees_u);
    w.i128(v.base_fee_units);
    write_inv(w, &v.pos);
    w.decimal(v.realized);
    w.option_u64(v.last_mark2);
    w.i128(v.peak_h);
    w.i128(v.max_dd_h);
    w.u64(v.max_abs_pos);
}

fn read_ledger(r: &mut Reader<'_>) -> MMResult<VenueLedger> {
    Ok(VenueLedger {
        cash_u: r.i128()?,
        fixed_fees_u: r.i128()?,
        base_fee_units: r.i128()?,
        pos: read_inv(r)?,
        realized: r.decimal()?,
        last_mark2: r.option_u64()?,
        peak_h: r.i128()?,
        max_dd_h: r.i128()?,
        max_abs_pos: r.u64()?,
    })
}

fn read_status(r: &mut Reader<'_>) -> MMResult<Status> {
    let tag = r.u8()?;
    let code = r.u16()?;
    match (tag, code) {
        (0, 0) => Ok(Status::Running),
        (1, 1) => Ok(Status::Terminal(TermReason::EndOfData)),
        (2, 0) => Ok(Status::Truncated),
        (3, code) => HaltReason::from_code(code)
            .map(Status::Halted)
            .ok_or_else(|| err(format!("invalid halt reason {code}"))),
        (tag, code) => Err(err(format!("invalid status ({tag}, {code})"))),
    }
}

fn read_phase(r: &mut Reader<'_>) -> MMResult<Phase> {
    let code = r.u8()?;
    Phase::from_code(code).ok_or_else(|| err(format!("invalid phase {code}")))
}

fn write_rng(w: &mut Writer, rng: &KernelRng) {
    w.u64(rng.seed);
    w.u64(rng.state);
}

fn read_rng(r: &mut Reader<'_>, algo: u8) -> MMResult<KernelRng> {
    let seed = r.u64()?;
    let state = r.u64()?;
    if state == 0 {
        return Err(err("random stream state is zero"));
    }
    Ok(KernelRng { algo, seed, state })
}

fn write_collection<T: Blank, const N: usize>(
    w: &mut Writer,
    a: &Arr<T, N>,
    item: impl Fn(&mut Writer, &T),
) {
    w.len(N);
    w.len(a.len());
    for value in a.as_slice() {
        item(w, value);
    }
}

/// Canonical encoding v1 of `s`.
#[must_use]
pub fn encode<const ORD: usize, const MSG: usize, const DLV: usize, const OBS: usize>(
    s: &KernelState<ORD, MSG, DLV, OBS>,
) -> Vec<u8> {
    let mut w = Writer::new();
    w.bytes(&STATE_MAGIC);
    w.u16(s.format_version);
    w.u8(s.rng_latency.algo);
    w.u8(s.rng_fill.algo);
    for cap in [ORD, MSG, DLV, OBS] {
        w.len(cap);
    }
    let id = &s.data_id;
    w.u16(id.scheme);
    w.u64(id.len);
    w.u64(id.first_ts);
    w.u64(id.last_ts);
    w.u128(id.digest);
    w.bytes(&id.external);
    w.u128(s.cfg_fp);
    w.u8(s.phase.code());
    w.u8(s.resume.code());
    w.u64(s.cursor);
    w.option_u64(s.cur_row);
    w.u64(s.now_ns);
    w.u64(s.step);
    write_status(&mut w, s.status);
    write_rng(&mut w, &s.rng_latency);
    write_rng(&mut w, &s.rng_fill);
    for v in [
        s.stamp_seq,
        s.next_msg_seq,
        s.next_order_id,
        s.next_exec_id,
        s.last_delivery_ns,
        s.last_observed_ns,
    ] {
        w.u64(v);
    }
    write_collection(&mut w, &s.orders, write_order);
    write_collection(&mut w, &s.msgs, |w, m| {
        w.u64(m.eff_ns);
        w.u64(m.seq);
        // Tags 3 and 4 only appear in runs with expiring quotes: a run
        // without them keeps the bytes of tags 1 and 2.
        match &m.kind {
            MsgKind::Activate { order, life: None } => {
                w.u8(1);
                write_order(w, order);
            }
            MsgKind::Cancel { id } => {
                w.u8(2);
                w.u64(*id);
            }
            MsgKind::Expire { id } => {
                w.u8(3);
                w.u64(*id);
            }
            MsgKind::Activate {
                order,
                life: Some(life),
            } => {
                w.u8(4);
                write_order(w, order);
                w.u64(life.get());
            }
        }
    });
    write_collection(&mut w, &s.awaiting, write_order);
    write_ledger(&mut w, &s.venue);
    w.i64(s.gate.venue_pos_lots);
    w.u64(s.gate.live_buy_lots);
    w.u64(s.gate.live_sell_lots);
    w.option_u64(s.gate.mark2);
    write_collection(&mut w, &s.deliveries, |w, d| {
        w.u64(d.at_ns);
        write_report(w, &d.report);
        match &d.fill {
            None => w.u8(0),
            Some(fill) => {
                w.u8(1);
                write_fill(w, fill);
            }
        }
    });
    write_collection(&mut w, &s.observed, |w, o| {
        w.u64(o.ready_ns);
        w.u64(o.row);
    });
    write_inv(&mut w, &s.agent.pos);
    w.option_u64(s.agent.last_obs_row);
    w.u64(s.agent.delivered_local_rejects);
    w.u64(s.agent.delivered_fills);
    write_collection(&mut w, &s.account, |w, o| {
        w.u64(o.order_id);
        w.bool(o.awaiting_cancel_outcome);
    });
    match s.last {
        None => {
            w.u8(0);
            w.u16(0);
        }
        Some(ActionStatus::Applied) => {
            w.u8(1);
            w.u16(0);
        }
        Some(ActionStatus::Rejected(reason)) => {
            w.u8(2);
            w.u16(reason.code());
        }
        Some(ActionStatus::QuoteError(code)) => {
            w.u8(3);
            w.u16(code.0);
        }
    }
    let c = &s.counters;
    w.u64(c.actions);
    for v in c.rejected {
        w.u64(v);
    }
    for v in [
        c.halts,
        c.eof_dropped_msgs,
        c.eof_dropped_obs,
        c.quote_errors,
        c.venue_fills,
        c.local_rejects,
    ] {
        w.u64(v);
    }
    w.finish()
}

/// FNV-1a-128 of the canonical encoding of `s`.
#[must_use]
pub fn fingerprint<const ORD: usize, const MSG: usize, const DLV: usize, const OBS: usize>(
    s: &KernelState<ORD, MSG, DLV, OBS>,
) -> u128 {
    fnv1a128(&encode(s))
}

/// Decodes a state for `kernel`, rejecting any state that breaks an
/// invariant `step` relies on.
///
/// The guarantee: an accepted state satisfies every invariant `step`
/// relies on (grid ranges, queue and capacity bounds, counter and sequence
/// relations, time clamps, and the ledger relations behind the exactness
/// bound), so `step` never halts or panics from it. Decoding does not prove
/// the state was reached by `reset` and `step`: a mutated encoding that
/// keeps every invariant can be accepted.
///
/// # Errors
///
/// Returns [`MMError::InvalidMarketState`] for a wrong magic, version,
/// capacity, dataset or configuration identity, an invalid code, a
/// truncated or oversized input, or a state that breaks an invariant
/// (sorted unique keys, gate sums, account coverage and queue bounds,
/// ledger bounds, phase and status consistency, sequence and id
/// monotonicity, FIFO times).
pub fn decode<const ORD: usize, const MSG: usize, const DLV: usize, const OBS: usize>(
    bytes: &[u8],
    kernel: &FinancialKernel<'_, ORD, MSG, DLV, OBS>,
) -> MMResult<KernelState<ORD, MSG, DLV, OBS>> {
    let mut r = Reader { bytes, pos: 0 };
    if r.arr::<4>()? != STATE_MAGIC {
        return Err(err("bad magic"));
    }
    let format_version = r.u16()?;
    if format_version != FORMAT_VERSION {
        return Err(err(format!("unsupported format {format_version}")));
    }
    let algo_latency = r.u8()?;
    let algo_fill = r.u8()?;
    if algo_latency != RNG_ALGO_XORSHIFT64STAR || algo_fill != RNG_ALGO_XORSHIFT64STAR {
        return Err(err(format!(
            "unsupported random algorithms ({algo_latency}, {algo_fill})"
        )));
    }
    for (name, cap) in [("ORD", ORD), ("MSG", MSG), ("DLV", DLV), ("OBS", OBS)] {
        let got = r.u32()?;
        if usize::try_from(got).ok() != Some(cap) {
            return Err(err(format!("capacity {name} {got} differs from {cap}")));
        }
    }
    let data_id = DatasetId {
        scheme: r.u16()?,
        len: r.u64()?,
        first_ts: r.u64()?,
        last_ts: r.u64()?,
        digest: r.u128()?,
        external: r.arr::<32>()?,
    };
    let cfg_fp = r.u128()?;
    let phase = read_phase(&mut r)?;
    let resume = read_phase(&mut r)?;
    let cursor = r.u64()?;
    let cur_row = r.option_u64()?;
    let now_ns = r.u64()?;
    let step = r.u64()?;
    let status = read_status(&mut r)?;
    let rng_latency = read_rng(&mut r, algo_latency)?;
    let rng_fill = read_rng(&mut r, algo_fill)?;
    let stamp_seq = r.u64()?;
    let next_msg_seq = r.u64()?;
    let next_order_id = r.u64()?;
    let next_exec_id = r.u64()?;
    let last_delivery_ns = r.u64()?;
    let last_observed_ns = r.u64()?;
    let orders = r.collection::<VenueOrder, ORD>("orders", read_order)?;
    let msgs = r.collection::<VenueMsg, MSG>("messages", |r| {
        let eff_ns = r.u64()?;
        let seq = r.u64()?;
        let kind = match r.u8()? {
            1 => MsgKind::Activate {
                order: read_order(r)?,
                life: None,
            },
            2 => MsgKind::Cancel { id: r.u64()? },
            3 => MsgKind::Expire { id: r.u64()? },
            4 => {
                let order = read_order(r)?;
                let life = NonZeroU64::new(r.u64()?)
                    .ok_or_else(|| err("expiring activation with a life of 0".to_string()))?;
                MsgKind::Activate {
                    order,
                    life: Some(life),
                }
            }
            tag => return Err(err(format!("invalid message tag {tag}"))),
        };
        Ok(VenueMsg { eff_ns, seq, kind })
    })?;
    let awaiting = r.collection::<VenueOrder, ORD>("awaiting", read_order)?;
    let venue = read_ledger(&mut r)?;
    let gate = GateState {
        venue_pos_lots: r.i64()?,
        live_buy_lots: r.u64()?,
        live_sell_lots: r.u64()?,
        mark2: r.option_u64()?,
    };
    let deliveries = r.collection::<Delivery, DLV>("deliveries", |r| {
        let at_ns = r.u64()?;
        let report = read_report(r)?;
        let fill = if r.bool()? { Some(read_fill(r)?) } else { None };
        Ok(Delivery {
            at_ns,
            report,
            fill,
        })
    })?;
    let observed = r.collection::<ObsEntry, OBS>("observations", |r| {
        Ok(ObsEntry {
            ready_ns: r.u64()?,
            row: r.u64()?,
        })
    })?;
    let agent = AgentView {
        pos: read_inv(&mut r)?,
        last_obs_row: r.option_u64()?,
        delivered_local_rejects: r.u64()?,
        delivered_fills: r.u64()?,
    };
    let account = r.collection::<OpenOrder, ORD>("account", |r| {
        Ok(OpenOrder {
            order_id: r.u64()?,
            awaiting_cancel_outcome: r.bool()?,
        })
    })?;
    let last = match (r.u8()?, r.u16()?) {
        (0, 0) => None,
        (1, 0) => Some(ActionStatus::Applied),
        (2, code) => Some(ActionStatus::Rejected(
            RejectReason::from_code(code)
                .ok_or_else(|| err(format!("invalid reject reason {code}")))?,
        )),
        (3, code) => Some(ActionStatus::QuoteError(super::state::ErrCode(code))),
        (tag, code) => return Err(err(format!("invalid last outcome ({tag}, {code})"))),
    };
    let actions = r.u64()?;
    let mut rejected = [0u64; 8];
    for slot in &mut rejected {
        *slot = r.u64()?;
    }
    let counters = Counters {
        actions,
        rejected,
        halts: r.u64()?,
        eof_dropped_msgs: r.u64()?,
        eof_dropped_obs: r.u64()?,
        quote_errors: r.u64()?,
        venue_fills: r.u64()?,
        local_rejects: r.u64()?,
    };
    if r.pos != bytes.len() {
        return Err(err(format!(
            "{} trailing bytes",
            bytes.len().saturating_sub(r.pos)
        )));
    }
    let s = KernelState {
        data_id,
        cfg_fp,
        format_version,
        phase,
        resume,
        cursor,
        cur_row,
        now_ns,
        step,
        status,
        rng_latency,
        rng_fill,
        stamp_seq,
        next_msg_seq,
        next_order_id,
        next_exec_id,
        last_delivery_ns,
        last_observed_ns,
        orders,
        msgs,
        awaiting,
        venue,
        gate,
        deliveries,
        observed,
        agent,
        account,
        last,
        counters,
    };
    validate_state(&s, kernel)?;
    Ok(s)
}

fn check(ok: bool, message: &str) -> MMResult<()> {
    if ok {
        Ok(())
    } else {
        Err(err(message.to_string()))
    }
}

fn strictly_increasing(values: impl Iterator<Item = u64>) -> bool {
    let mut previous: Option<u64> = None;
    for v in values {
        if previous.is_some_and(|p| p >= v) {
            return false;
        }
        previous = Some(v);
    }
    true
}

fn non_decreasing(values: impl Iterator<Item = u64>) -> bool {
    let mut previous: Option<u64> = None;
    for v in values {
        if previous.is_some_and(|p| p > v) {
            return false;
        }
        previous = Some(v);
    }
    true
}

#[allow(clippy::too_many_lines)] // One linear list of invariants.
fn validate_state<const ORD: usize, const MSG: usize, const DLV: usize, const OBS: usize>(
    s: &KernelState<ORD, MSG, DLV, OBS>,
    k: &FinancialKernel<'_, ORD, MSG, DLV, OBS>,
) -> MMResult<()> {
    let cfg = &k.cfg;
    check(
        s.data_id == k.data.id,
        "dataset identity differs from the kernel",
    )?;
    check(
        s.cfg_fp == cfg.fingerprint,
        "configuration fingerprint differs from the kernel",
    )?;
    // Rows and phase.
    let len = k.data.id.len;
    check(s.cursor <= len, "cursor beyond the dataset")?;
    match s.cur_row {
        None => check(s.cursor == 0, "no current row after reading rows")?,
        Some(row) => check(
            row.checked_add(1) == Some(s.cursor),
            "current row is not the last row read",
        )?,
    }
    check(
        matches!(s.resume, Phase::BeforeMatch | Phase::ObsQueued),
        "resume phase must be BeforeMatch or ObsQueued",
    )?;
    check(
        (s.phase == Phase::Terminal) == !s.status.is_running(),
        "phase Terminal iff status is not running",
    )?;
    if s.phase == Phase::BeforeMatch
        || (s.phase == Phase::AwaitDecision && s.resume == Phase::BeforeMatch)
    {
        check(s.cur_row.is_some(), "BeforeMatch without a current row")?;
    }
    if s.phase == Phase::AwaitDecision {
        check(
            s.agent.last_obs_row.is_some(),
            "decision point without an observation",
        )?;
    }
    check(s.step <= cfg.s_max, "step beyond max_steps")?;
    check(
        s.counters.actions == s.step,
        "action counter differs from the step count",
    )?;
    let rejected = s
        .counters
        .rejected
        .iter()
        .try_fold(0u64, |a, v| a.checked_add(*v));
    check(
        rejected.is_some_and(|r| r <= s.step),
        "reject counters exceed the steps",
    )?;
    check(
        s.counters.quote_errors <= s.step,
        "quote errors exceed the steps",
    )?;
    check(
        s.counters.halts == u64::from(matches!(s.status, Status::Halted(_))),
        "halt counter inconsistent with the status",
    )?;
    // Observation FIFO.
    let obs = s.observed.as_slice();
    check(
        strictly_increasing(obs.iter().map(|o| o.row)),
        "observation rows not increasing",
    )?;
    check(
        non_decreasing(obs.iter().map(|o| o.ready_ns)),
        "observation times decrease",
    )?;
    check(
        obs.iter().all(|o| o.ready_ns <= s.last_observed_ns),
        "observation after the last ready time",
    )?;
    // Before matching, the current row's observation is not queued yet.
    let before_match = match s.phase {
        Phase::BeforeMatch => true,
        Phase::AwaitDecision => s.resume == Phase::BeforeMatch,
        Phase::ObsQueued | Phase::Terminal => false,
    };
    let row_limit = if before_match {
        s.cur_row
    } else {
        s.cur_row.and_then(|r| r.checked_add(1))
    };
    check(
        obs.iter().all(|o| row_limit.is_some_and(|l| o.row < l)),
        "observation of a row not yet queued",
    )?;
    if let Some(last) = s.agent.last_obs_row {
        check(
            obs.iter().all(|o| o.row > last),
            "queued observation already observed",
        )?;
        check(s.cursor > last, "observed row not read")?;
    }
    // Deliveries.
    let dl = s.deliveries.as_slice();
    check(
        non_decreasing(dl.iter().map(|d| d.at_ns)),
        "delivery times decrease",
    )?;
    check(
        dl.iter().all(|d| d.at_ns <= s.last_delivery_ns),
        "delivery after the last delivery time",
    )?;
    check(
        dl.iter().all(|d| d.report.seq < s.stamp_seq),
        "report sequence not below stamp_seq",
    )?;
    check(
        dl.iter()
            .all(|d| d.fill.is_none_or(|f| f.exec_id < s.next_exec_id)),
        "execution id not below next_exec_id",
    )?;
    check(
        dl.iter().all(|d| d.report.order_id < s.next_order_id),
        "report order id not below next_order_id",
    )?;
    // Venue messages and orders.
    let msgs = s.msgs.as_slice();
    for pair in msgs.windows(2) {
        if let [a, b] = pair {
            check(a.key() < b.key(), "messages not sorted by (eff, seq)")?;
        }
    }
    check(
        msgs.iter().all(|m| m.seq < s.next_msg_seq),
        "message sequence not below next_msg_seq",
    )?;
    let activations: Vec<VenueOrder> = msgs
        .iter()
        .filter_map(|m| match m.kind {
            MsgKind::Activate { order, .. } => Some(order),
            MsgKind::Cancel { .. } | MsgKind::Expire { .. } => None,
        })
        .collect();
    let cancels: Vec<u64> = msgs
        .iter()
        .filter_map(|m| match m.kind {
            MsgKind::Cancel { id } => Some(id),
            MsgKind::Activate { .. } | MsgKind::Expire { .. } => None,
        })
        .collect();
    for m in msgs {
        if let MsgKind::Activate { order, .. } = m.kind {
            check(
                order.state == OrderState::Submitted
                    && order.active_after_ns == m.eff_ns
                    && order.filled == 0,
                "activation not a fresh submitted order",
            )?;
        }
    }
    check(
        s.orders
            .as_slice()
            .iter()
            .all(|o| o.state == OrderState::Active),
        "resting order not active",
    )?;
    // An expiry names a resting order, after its activation, at most once.
    // With an activation per pending order (unique ids) and a cancel per
    // flagged order (the flags check below), no order then holds more than its
    // two venue messages: an activation and an expiry never name the same
    // order.
    let mut expiring: Vec<u64> = Vec::new();
    for m in msgs {
        if let MsgKind::Expire { id } = m.kind {
            check(
                s.orders
                    .as_slice()
                    .iter()
                    .any(|o| o.id == id && m.eff_ns > o.active_after_ns),
                "expiry not for a resting order after its activation",
            )?;
            expiring.push(id);
        }
    }
    expiring.sort_unstable();
    check(
        strictly_increasing(expiring.iter().copied()),
        "duplicate expiry",
    )?;
    // An expired order with a cancel in flight awaits it, as a filled one.
    check(
        s.awaiting.as_slice().iter().all(|o| {
            o.cancel_inflight
                && matches!(
                    o.state,
                    OrderState::Filled | OrderState::Rejected | OrderState::Canceled
                )
        }),
        "awaiting order not terminal with a cancel in flight",
    )?;
    check(
        strictly_increasing(s.awaiting.as_slice().iter().map(|o| o.id)),
        "awaiting ids not sorted",
    )?;
    let all: Vec<&VenueOrder> = s
        .orders
        .as_slice()
        .iter()
        .chain(activations.iter())
        .chain(s.awaiting.as_slice().iter())
        .collect();
    let mut ids: Vec<u64> = all.iter().map(|o| o.id).collect();
    ids.sort_unstable();
    check(
        strictly_increasing(ids.iter().copied()),
        "duplicate venue order ids",
    )?;
    check(
        all.iter().all(|o| o.id >= 1 && o.id < s.next_order_id),
        "venue order id not below next_order_id",
    )?;
    check(
        all.iter().all(|o| o.filled <= o.qty),
        "order filled beyond its quantity",
    )?;
    check(
        cfg.fill != ValidFill::Immediate || s.orders.as_slice().iter().all(|o| o.filled == 0),
        "resting order partially filled under the immediate model",
    )?;
    let flagged: Vec<u64> = {
        let mut v: Vec<u64> = all
            .iter()
            .filter(|o| o.cancel_inflight)
            .map(|o| o.id)
            .collect();
        v.sort_unstable();
        v
    };
    let mut sorted_cancels = cancels.clone();
    sorted_cancels.sort_unstable();
    if s.status == Status::Terminal(TermReason::EndOfData) {
        // The end of the data delivers every report and drops the pending
        // messages and observations; cancel flags may remain.
        check(
            msgs.is_empty() && s.observed.is_empty() && s.deliveries.is_empty(),
            "queues not drained at the end of the data",
        )?;
    } else {
        check(
            flagged == sorted_cancels,
            "cancel messages and cancel flags disagree",
        )?;
    }
    // Gate sums.
    let live = |dir: Dir| {
        s.orders
            .as_slice()
            .iter()
            .chain(activations.iter())
            .filter(|o| o.side == dir)
            .try_fold(0u64, |a, o| a.checked_add(o.remaining()))
    };
    // At the end of the data dropped activations keep their reservation,
    // as in the engine's policy.
    let eof = s.status == Status::Terminal(TermReason::EndOfData);
    let reserved =
        |sum: Option<u64>, gate: u64| sum.is_some_and(|v| if eof { v <= gate } else { v == gate });
    check(
        reserved(live(Dir::Buy), s.gate.live_buy_lots),
        "gate buy reservation differs from the orders",
    )?;
    check(
        reserved(live(Dir::Sell), s.gate.live_sell_lots),
        "gate sell reservation differs from the orders",
    )?;
    // Account coverage and queue bounds.
    let account = s.account.as_slice();
    check(
        strictly_increasing(account.iter().map(|o| o.order_id)),
        "account ids not sorted",
    )?;
    check(
        account.iter().all(|o| o.order_id < s.next_order_id),
        "account id not below next_order_id",
    )?;
    let open = |id: u64| account.iter().any(|o| o.order_id == id);
    check(
        all.iter().all(|o| open(o.id)),
        "venue order not open in the account",
    )?;
    check(
        cancels.iter().all(|id| open(*id)),
        "cancel for an order not open in the account",
    )?;
    check(
        dl.iter().all(|d| open(d.report.order_id)),
        "delivery for an order not open in the account",
    )?;
    let held = account.len();
    let resting = s.orders.len().saturating_add(s.awaiting.len());
    check(resting <= held, "more venue orders than open orders")?;
    check(
        msgs.len() <= held.saturating_mul(2),
        "more messages than twice the open orders",
    )?;
    // A report in flight belongs to an open order, which reports at most once
    // per row on top of its three status reports.
    let per_order = usize::try_from(cfg.partial_reports_per_order)
        .unwrap_or(usize::MAX)
        .saturating_add(3);
    check(
        dl.len() <= held.saturating_mul(per_order),
        "more deliveries than the open orders can have in flight",
    )?;
    // Ledger.
    let v = &s.venue;
    let b = cfg.bound_b;
    for (name, value) in [
        ("cash", v.cash_u),
        ("fees", v.fixed_fees_u),
        ("base fees", v.base_fee_units),
    ] {
        check(
            value.unsigned_abs() <= b,
            &format!("{name} beyond the bound"),
        )?;
    }
    check(v.base_fee_units == 0, "base fees in a quote fee kernel")?;
    {
        let (name, value) = ("peak", v.peak_h);
        check(
            value.unsigned_abs() <= cfg.half_bound(),
            &format!("{name} beyond the half-U bound"),
        )?;
    }
    check(v.max_dd_h >= 0, "negative drawdown")?;
    check(
        scaled_decimal(v.cash_u, cfg.s) == Some(v.realized),
        "realized differs from the cash shadow",
    )?;
    let pos = cfg.lot.to_decimal(i128::from(s.gate.venue_pos_lots));
    check(
        pos == Some(v.pos.quantity),
        "venue position differs from the gate lots",
    )?;
    check(
        s.gate.venue_pos_lots.unsigned_abs() <= cfg.x_eff,
        "venue position beyond the position bound",
    )?;
    check(
        s.gate.venue_pos_lots.unsigned_abs() <= v.max_abs_pos,
        "position above its recorded maximum",
    )?;
    check(
        v.last_mark2.is_none_or(|m| m <= cfg.p.saturating_mul(2)),
        "mark beyond the price range",
    )?;
    check(
        s.next_exec_id >= 1 && s.next_order_id >= 1,
        "ids start at 1",
    )?;
    check(
        s.counters.venue_fills.checked_add(1) == Some(s.next_exec_id),
        "fill counter differs from the execution ids",
    )?;
    let decimal_ok = |d: Decimal| d.scale() <= 28;
    check(
        decimal_ok(v.realized) && decimal_ok(s.agent.pos.quantity),
        "decimal scale beyond 28",
    )?;
    validate_invariants(s, k, &activations, &cancels)
}

/// Invariants `step` relies on: grid ranges, agent view
/// against the venue counters, account coverage, step and time limits.
#[allow(clippy::too_many_lines)] // One linear list of invariants.
fn validate_invariants<const ORD: usize, const MSG: usize, const DLV: usize, const OBS: usize>(
    s: &KernelState<ORD, MSG, DLV, OBS>,
    k: &FinancialKernel<'_, ORD, MSG, DLV, OBS>,
    activations: &[VenueOrder],
    cancels: &[u64],
) -> MMResult<()> {
    let cfg = &k.cfg;
    let price_ok = |p: u64| p >= cfg.min_price && p <= cfg.p;
    let qty_ok = |q: u64| q >= 1 && q <= cfg.q;
    // Grid ranges of every order, report and fill.
    let venue = s
        .orders
        .as_slice()
        .iter()
        .chain(activations.iter())
        .chain(s.awaiting.as_slice().iter());
    for o in venue {
        check(price_ok(o.price), "order price outside the price range")?;
        check(qty_ok(o.qty), "order quantity outside the size range")?;
    }
    let dl = s.deliveries.as_slice();
    check_report_chains(s, cfg)?;
    for d in dl {
        let r = &d.report;
        check(price_ok(r.price), "report price outside the price range")?;
        check(
            qty_ok(r.qty) && r.last <= r.qty && r.cum <= r.qty,
            "report quantity outside the size range",
        )?;
        match (r.kind, &d.fill) {
            (ExecKind::Fill | ExecKind::PartialFill, Some(f)) => {
                check(
                    price_ok(f.price) && qty_ok(f.qty),
                    "fill outside the grid ranges",
                )?;
                check(
                    f.order_id == r.order_id
                        && f.side == r.side
                        && f.qty == r.last
                        && f.price == r.price,
                    "fill differs from its report",
                )?;
            }
            (ExecKind::Fill | ExecKind::PartialFill, None) => {
                return Err(err("fill report without a fill"));
            }
            (
                ExecKind::Accepted
                | ExecKind::Rejected
                | ExecKind::Canceled
                | ExecKind::CancelRejected,
                Some(_),
            ) => return Err(err("fill attached to a status report")),
            (
                ExecKind::Accepted
                | ExecKind::Rejected
                | ExecKind::Canceled
                | ExecKind::CancelRejected,
                None,
            ) => {}
        }
        if r.origin == Origin::Local {
            check(
                r.kind == ExecKind::Rejected && r.cum == 0,
                "local report that is not a reject",
            )?;
        }
    }
    // The agent view is the venue minus the queued deliveries.
    let queued_fills = dl.iter().filter(|d| d.fill.is_some()).count();
    let queued_local = dl
        .iter()
        .filter(|d| d.report.kind == ExecKind::Rejected && d.report.origin == Origin::Local)
        .count();
    let as_u64 = |n: usize| u64::try_from(n).ok();
    check(
        as_u64(queued_fills).and_then(|q| q.checked_add(s.agent.delivered_fills))
            == Some(s.counters.venue_fills),
        "delivered fills differ from the venue fills",
    )?;
    check(
        as_u64(queued_local).and_then(|q| q.checked_add(s.agent.delivered_local_rejects))
            == Some(s.counters.local_rejects),
        "delivered local rejects differ from the local rejects",
    )?;
    let queued_lots = dl.iter().filter_map(|d| d.fill).try_fold(0i128, |acc, f| {
        let lots = i128::from(f.qty);
        match Dir::try_from(f.side).ok()? {
            Dir::Buy => acc.checked_add(lots),
            Dir::Sell => acc.checked_sub(lots),
        }
    });
    let agent_lots = queued_lots.and_then(|q| i128::from(s.gate.venue_pos_lots).checked_sub(q));
    check(
        agent_lots.and_then(|l| cfg.lot.to_decimal(l)) == Some(s.agent.pos.quantity),
        "agent position differs from the venue position minus queued fills",
    )?;
    for pos in [&s.agent.pos, &s.venue.pos] {
        check(
            !pos.quantity.is_zero() || pos.avg_entry.is_zero(),
            "flat position with an entry price",
        )?;
    }
    // A position's average entry is a weighted mean of fill prices, so it
    // lies in the price range.
    let lowest = cfg.tick.to_decimal(i128::from(cfg.min_price.max(1)));
    let highest = cfg.tick.to_decimal(i128::from(cfg.p));
    for pos in [&s.agent.pos, &s.venue.pos] {
        if !pos.quantity.is_zero() {
            let inside = lowest.is_some_and(|l| pos.avg_entry >= l)
                && highest.is_some_and(|h| pos.avg_entry <= h);
            check(inside, "average entry outside the price range")?;
        }
    }
    // Every open order has a counterpart at the venue or in the deliveries
    // (at the end of the data dropped activations leave none).
    if s.status != Status::Terminal(TermReason::EndOfData) {
        for open in s.account.as_slice() {
            let id = open.order_id;
            let found = s.orders.as_slice().iter().any(|o| o.id == id)
                || activations.iter().any(|o| o.id == id)
                || s.awaiting.as_slice().iter().any(|o| o.id == id)
                || cancels.contains(&id)
                || dl.iter().any(|d| d.report.order_id == id);
            check(found, "open order without a venue or delivery counterpart")?;
        }
    }
    // Step limit: a running state waits at a decision below max_steps.
    if s.status.is_running() {
        check(
            s.phase == Phase::AwaitDecision && s.step < cfg.s_max,
            "running state not at a decision below max_steps",
        )?;
    }
    check(
        s.venue.max_abs_pos <= cfg.x_eff,
        "maximum position beyond the position bound",
    )?;
    validate_counters(s, k)?;
    validate_ledger(s, k)?;
    // Time: between the previous and the current row.
    match s.cur_row {
        None => check(s.now_ns == 0, "time advanced before the first row")?,
        Some(row) => {
            let ts = |r: u64| k.data.row(r).map(|t| t.ts_ns);
            let current = ts(row).ok_or_else(|| err("current row outside the dataset"))?;
            let previous = match row.checked_sub(1) {
                Some(r) => ts(r).ok_or_else(|| err("previous row outside the dataset"))?,
                None => 0,
            };
            check(
                previous <= s.now_ns && s.now_ns <= current,
                "time outside the current row window",
            )?;
            let after_match = match s.phase {
                Phase::ObsQueued => true,
                Phase::AwaitDecision => s.resume == Phase::ObsQueued,
                Phase::BeforeMatch => false,
                Phase::Terminal => s.status == Status::Terminal(TermReason::EndOfData),
            };
            if after_match {
                check(
                    s.now_ns == current,
                    "time differs from the current row after matching",
                )?;
            }
        }
    }
    Ok(())
}

/// The queued reports of each order against each other and the order: every
/// fill books lots and raises `cum` by exactly its `last`, a `PartialFill`
/// leaves quantity open and a `Fill` none, a `Canceled` keeps what was filled
/// and the last queued `cum` of a venue order is its `filled`. The immediate
/// model has no partial fill, so nothing is filled before a `Fill`.
fn check_report_chains<const ORD: usize, const MSG: usize, const DLV: usize, const OBS: usize>(
    s: &KernelState<ORD, MSG, DLV, OBS>,
    cfg: &ValidatedConfig,
) -> MMResult<()> {
    let immediate = cfg.fill == ValidFill::Immediate;
    let mut last_cum: BTreeMap<u64, u64> = BTreeMap::new();
    for d in s.deliveries.as_slice() {
        let r = &d.report;
        match r.kind {
            ExecKind::PartialFill | ExecKind::Fill => {
                check(r.last > 0, "fill report without a quantity")?;
                let open = r.kind == ExecKind::PartialFill;
                check(
                    !(open && immediate),
                    "partial fill report under the immediate model",
                )?;
                check(
                    if open { r.cum < r.qty } else { r.cum == r.qty },
                    "fill report kind differs from its open quantity",
                )?;
                check(
                    !immediate || r.last == r.cum,
                    "fill after a partial fill under the immediate model",
                )?;
                match last_cum.get(&r.order_id) {
                    Some(prior) => check(
                        prior.checked_add(r.last) == Some(r.cum),
                        "fill reports of an order do not add up",
                    )?,
                    None => check(r.last <= r.cum, "fill beyond the cumulative quantity")?,
                }
                last_cum.insert(r.order_id, r.cum);
            }
            ExecKind::Canceled => {
                check(
                    !immediate || r.cum == 0,
                    "filled quantity on a cancel under the immediate model",
                )?;
                check(r.cum < r.qty, "cancel of a filled order")?;
                if let Some(prior) = last_cum.get(&r.order_id) {
                    check(*prior == r.cum, "cancel report differs from the fills")?;
                }
            }
            ExecKind::Accepted | ExecKind::Rejected | ExecKind::CancelRejected => {}
        }
    }
    for o in s.orders.as_slice().iter().chain(s.awaiting.as_slice()) {
        if let Some(cum) = last_cum.get(&o.id) {
            check(*cum == o.filled, "order filled differs from its reports")?;
        }
    }
    Ok(())
}

/// Counters and sequences against their call-site relations, and the time
/// clamps against the clock and the rows.
///
/// With `created = next_order_id - 1` orders (each admitted quote creates
/// two, a one-sided action one):
/// - `created <= 2 * step`;
/// - stamps: an order takes one at submission, a local reject a second one,
///   a venue order at most an activation, a cancel outcome and one per fill,
///   so `created + local_rejects + venue_fills <= stamp_seq`, and
///   `stamp_seq <= 4 * created` for the immediate model, with its one fill per
///   order, or `4 * created + venue_fills` for the queue position model (an
///   expiring order can take an expiry's cancel and then the agent's
///   rejected cancel);
/// - messages: an activation per order not rejected locally, at most one
///   cancel and at most one expiry per order, so
///   `next_msg_seq <= 3 * created - local_rejects`;
/// - `local_rejects + venue_fills <= created * fills per order` (one for the
///   immediate model, the largest order for the queue position model, since a
///   fill books at least one lot), `quote_errors` equals the
///   `QuoteError` reject count, and the end-of-data drops are zero before
///   the end and bounded by the messages and rows;
/// - at a decision every venue message and delivery due by `now_ns` was
///   applied, and every scheduled time is within the largest delay of it.
#[allow(clippy::too_many_lines)]
// One linear list of relations.
// Every operand is a `u64` widened to `u128`; at most three are summed or
// one is multiplied by 4, which stays far below `u128::MAX`.
#[allow(clippy::arithmetic_side_effects)]
fn validate_counters<const ORD: usize, const MSG: usize, const DLV: usize, const OBS: usize>(
    s: &KernelState<ORD, MSG, DLV, OBS>,
    k: &FinancialKernel<'_, ORD, MSG, DLV, OBS>,
) -> MMResult<()> {
    let cfg = &k.cfg;
    let c = &s.counters;
    let created = s
        .next_order_id
        .checked_sub(1)
        .ok_or_else(|| err("next_order_id below 1"))?;
    let wide = u128::from;
    check(
        wide(created) <= wide(s.step) * 2,
        "more orders than two per step",
    )?;
    let floor = wide(created) + wide(c.local_rejects) + wide(c.venue_fills);
    check(
        floor <= wide(s.stamp_seq)
            && wide(s.stamp_seq)
                <= match cfg.fill {
                    ValidFill::Immediate => wide(created) * 4,
                    ValidFill::QueuePosition { .. } => wide(created) * 4 + wide(c.venue_fills),
                },
        "stamp_seq outside its call-site bounds",
    )?;
    check(
        wide(s.next_msg_seq) + wide(c.local_rejects) <= wide(created) * 3,
        "next_msg_seq beyond three messages per order",
    )?;
    check(
        wide(c.local_rejects) + wide(c.venue_fills)
            <= wide(created)
                * match cfg.fill {
                    ValidFill::Immediate => 1,
                    ValidFill::QueuePosition { .. } => wide(cfg.q),
                },
        "more local rejects and fills than orders",
    )?;
    let quote_rejects = c.rejected.get(RejectReason::QuoteError.index()).copied();
    check(
        quote_rejects == Some(c.quote_errors),
        "quote errors differ from their reject count",
    )?;
    let eof = s.status == Status::Terminal(TermReason::EndOfData);
    if eof {
        check(
            wide(c.eof_dropped_msgs) <= wide(created) * 2 && c.eof_dropped_obs <= k.data.id.len,
            "end-of-data drops beyond the messages and rows",
        )?;
    } else {
        check(
            c.eof_dropped_msgs == 0 && c.eof_dropped_obs == 0,
            "drops before the end of the data",
        )?;
    }
    // Scheduled times.
    let lat = &cfg.latency;
    let span = |base: u64| {
        base.checked_add(lat.jitter_q)
            .and_then(|q| lat.ns(q))
            .ok_or_else(|| err("delay overflows"))
    };
    let venue_delay = span(lat.submit_q.max(lat.cancel_q))?;
    let ack_delay = span(lat.ack_q)?;
    let obs_delay = span(lat.observation_q)?;
    let obs_min = lat
        .ns(lat.observation_q)
        .ok_or_else(|| err("delay overflows"))?;
    let now = wide(s.now_ns);
    // An expiry is the order's own life after its activation, not a delay;
    // its time is checked against its order's activation.
    check(
        s.msgs.as_slice().iter().all(|m| {
            matches!(m.kind, MsgKind::Expire { .. }) || wide(m.eff_ns) <= now + wide(venue_delay)
        }),
        "venue message beyond the largest delay",
    )?;
    check(
        wide(s.last_delivery_ns) <= now + wide(ack_delay),
        "last delivery time beyond the acknowledgement delay",
    )?;
    if s.stamp_seq == 0 {
        check(s.last_delivery_ns == 0, "delivery time without any report")?;
    }
    // Every queued observation is within its row's delay, so the
    // observation capacity required by `new` covers it.
    for o in s.observed.as_slice() {
        let row = k
            .data
            .row(o.row)
            .ok_or_else(|| err("observation of a row outside the dataset"))?;
        let t = wide(o.ready_ns);
        check(
            wide(row.ts_ns) + wide(obs_min) <= t && t <= wide(row.ts_ns) + wide(obs_delay),
            "observation time outside its row's delay",
        )?;
    }
    if s.phase == Phase::AwaitDecision {
        check(
            s.observed.as_slice().iter().all(|o| o.ready_ns >= s.now_ns),
            "observation due before the decision not delivered",
        )?;
        check(
            s.msgs.as_slice().iter().all(|m| m.eff_ns > s.now_ns),
            "venue message due at the decision not applied",
        )?;
        check(
            s.deliveries.as_slice().iter().all(|d| d.at_ns > s.now_ns),
            "delivery due at the decision not applied",
        )?;
    }
    // The last observation time belongs to the last row queued.
    let queued_upto = match (s.phase, s.resume) {
        (Phase::BeforeMatch, _) | (Phase::AwaitDecision, Phase::BeforeMatch) => {
            s.cur_row.and_then(|r| r.checked_sub(1))
        }
        (Phase::ObsQueued | Phase::AwaitDecision | Phase::Terminal, _) => s.cur_row,
    };
    match queued_upto.and_then(|r| k.data.row(r)) {
        None => check(s.last_observed_ns == 0, "observation time before any row")?,
        Some(row) => {
            let low = wide(row.ts_ns) + wide(obs_min);
            let high = wide(row.ts_ns) + wide(obs_delay);
            let t = wide(s.last_observed_ns);
            check(
                low <= t && t <= high,
                "last observation time outside its row's delay",
            )?;
        }
    }
    Ok(())
}

/// The ledger against the fills that built it, so the bound argument of
/// `B` holds for every state `decode` accepts: with `f` venue fills (each at
/// most `P` ticks and `Q` lots), `|cash| <= f * N`, `|fees| <= f * Fee`
/// with the fee rate's sign and at least the smallest fee per fill,
/// `|position| <= f * Q`, the peak at least the initial equity and the
/// equity at the last mark within its bound. Step's cross-checks (cash
/// against realized, lots against the position, equity and drawdown within
/// their bounds) then cannot fail on this state or any later one.
fn validate_ledger<const ORD: usize, const MSG: usize, const DLV: usize, const OBS: usize>(
    s: &KernelState<ORD, MSG, DLV, OBS>,
    k: &FinancialKernel<'_, ORD, MSG, DLV, OBS>,
) -> MMResult<()> {
    let cfg = &k.cfg;
    let v = &s.venue;
    let fills = u128::from(s.counters.venue_fills);
    let f = cfg
        .fee_scale()
        .and_then(|f| u128::try_from(f).ok())
        .ok_or_else(|| err("fee scale"))?;
    // An upper bound that saturates at `u128::MAX` is still an upper bound;
    // a lower bound that overflows exceeds every value, so it rejects.
    let bound = |factors: &[u128]| factors.iter().fold(1u128, |acc, &x| acc.saturating_mul(x));
    let floor = |factors: &[u128], what: &str| {
        factors
            .iter()
            .try_fold(1u128, |acc, &x| acc.checked_mul(x))
            .ok_or_else(|| err(what))
    };
    let unit = [u128::from(cfg.tick.mantissa), u128::from(cfg.lot.mantissa)];
    let per_fill = bound(&[u128::from(cfg.p), u128::from(cfg.q), bound(&unit)]);
    let r = cfg.rate_num.unsigned_abs();
    check(
        v.cash_u.unsigned_abs() <= bound(&[fills, per_fill, f]),
        "cash beyond the fills' notional",
    )?;
    let fees = v.fixed_fees_u;
    check(
        fees.unsigned_abs() <= bound(&[fills, per_fill, r]),
        "fees beyond the fills' largest fees",
    )?;
    let [tick, lot] = unit;
    check(
        fees.unsigned_abs() >= floor(&[fills, tick, lot, r], "fee floor overflows")?,
        "fees below the fills' smallest fees",
    )?;
    let sign_ok = match cfg.rate_num.signum() {
        0 => fees == 0,
        1 => fees >= 0,
        _ => fees <= 0,
    };
    check(sign_ok, "fees with the wrong sign for the fee rate")?;
    let lots = bound(&[u128::from(cfg.q), fills]);
    check(
        u128::from(s.gate.venue_pos_lots.unsigned_abs()) <= lots
            && u128::from(v.max_abs_pos) <= lots,
        "position beyond the fills' quantity",
    )?;
    let initial = cfg
        .c0_u
        .checked_mul(2)
        .ok_or_else(|| err("initial equity"))?;
    check(v.peak_h >= initial, "peak below the initial equity")?;
    if let Some(mark2) = v.last_mark2 {
        let equity = super::step::equity_h(cfg, s, mark2).ok_or_else(|| err("equity overflows"))?;
        check(
            equity.unsigned_abs() <= cfg.half_bound(),
            "equity beyond the bound",
        )?;
    }
    check(
        v.max_dd_h.unsigned_abs() <= cfg.drawdown_bound(),
        "drawdown beyond the bound",
    )?;
    Ok(())
}

/// Inserts `order` keeping `awaiting` sorted by id.
pub(super) fn insert_by_id<const N: usize>(
    awaiting: &mut Arr<VenueOrder, N>,
    order: VenueOrder,
) -> Result<(), super::book::Full> {
    let index = awaiting
        .position(|o| o.id > order.id)
        .unwrap_or(awaiting.len());
    awaiting.insert(index, order)
}
