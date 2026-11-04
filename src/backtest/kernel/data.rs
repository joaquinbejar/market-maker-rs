//! Immutable, validated datasets on the kernel grid.

use crate::Decimal;
use crate::backtest::data::MarketTick;
use crate::types::error::{MMError, MMResult};

use super::config::KernelConfig;
use super::encode::Fnv128;
use super::grid::Unit;

/// Digest scheme 1: FNV-1a-128 over the canonical ticks.
pub const DIGEST_FNV1A128: u16 = 1;

/// Nanoseconds per millisecond, the unit of [`MarketTick::timestamp`].
pub(super) const NS_PER_MS: u64 = 1_000_000;

/// One market row on the kernel grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelTick {
    /// Exchange time, in nanoseconds.
    pub ts_ns: u64,
    /// Best bid, in ticks.
    pub bid: u64,
    /// Best bid size, in lots.
    pub bid_lots: u64,
    /// Best ask, in ticks.
    pub ask: u64,
    /// Best ask size, in lots.
    pub ask_lots: u64,
}

impl KernelTick {
    /// Twice the mid, in ticks (`bid + ask`).
    #[must_use]
    pub fn mid2(&self) -> Option<u64> {
        self.bid.checked_add(self.ask)
    }
}

/// Identity of a dataset.
///
/// `digest` identifies the data; it is not a security boundary. A bridge
/// crate may put its own cryptographic digest in `external`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DatasetId {
    /// Digest scheme ([`DIGEST_FNV1A128`]).
    pub scheme: u16,
    /// Number of rows.
    pub len: u64,
    /// First timestamp (0 when empty).
    pub first_ts: u64,
    /// Last timestamp (0 when empty).
    pub last_ts: u64,
    /// FNV-1a-128 over the grid and the canonical ticks.
    pub digest: u128,
    /// Optional caller supplied digest; zero when unused.
    pub external: [u8; 32],
}

/// Grid a dataset was built on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct DataGrid {
    pub(super) tick_m: u64,
    pub(super) tick_s: u32,
    pub(super) lot_m: u64,
    pub(super) lot_s: u32,
    pub(super) max_price: u64,
}

/// Immutable dataset with non-decreasing timestamps on the kernel grid.
#[derive(Debug, Clone, PartialEq)]
pub struct KernelDataset {
    pub(super) ticks: Vec<KernelTick>,
    pub(super) id: DatasetId,
    pub(super) grid: DataGrid,
}

fn data_error(row: usize, message: String) -> MMError {
    MMError::InvalidMarketState(format!("row {row}: {message}"))
}

impl KernelDataset {
    /// Converts engine ticks (timestamps in milliseconds) to the grid of
    /// `cfg`.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] if the grid of `cfg` is
    /// invalid, and [`MMError::InvalidMarketState`] for an off-grid price
    /// or size, a price or size written at a scale above the tick or lot
    /// scale (trailing zeros), a non-positive price, `bid >= ask`, a price above
    /// `max_price`, a decreasing timestamp, or a timestamp whose events
    /// could overflow `u64` nanoseconds.
    pub fn from_ticks(ticks: &[MarketTick], cfg: &KernelConfig) -> MMResult<Self> {
        let (tick, lot) = units(cfg)?;
        let mut rows = Vec::with_capacity(ticks.len());
        for (row, t) in ticks.iter().enumerate() {
            let ts_ns = t.timestamp.checked_mul(NS_PER_MS).ok_or_else(|| {
                data_error(row, format!("timestamp {} ms overflows ns", t.timestamp))
            })?;
            let price = |value: Decimal, what: &str| {
                // A raw scale above the grid scale would raise the engine's
                // mantissas above the bound; it is rejected.
                (value.scale() <= tick.scale)
                    .then(|| tick.count(value))
                    .flatten()
                    .and_then(|v| u64::try_from(v).ok())
                    .ok_or_else(|| {
                        data_error(row, format!("{what} {value} is not on the tick grid"))
                    })
            };
            let size = |value: Decimal, what: &str| {
                (value.scale() <= lot.scale)
                    .then(|| lot.count(value))
                    .flatten()
                    .and_then(|v| u64::try_from(v).ok())
                    .ok_or_else(|| {
                        data_error(row, format!("{what} {value} is not on the lot grid"))
                    })
            };
            rows.push(KernelTick {
                ts_ns,
                bid: price(t.bid_price, "bid")?,
                bid_lots: size(t.bid_size, "bid size")?,
                ask: price(t.ask_price, "ask")?,
                ask_lots: size(t.ask_size, "ask size")?,
            });
        }
        Self::from_kernel_ticks(rows, cfg)
    }

    /// Validates rows already on the grid of `cfg`.
    ///
    /// # Errors
    ///
    /// As [`KernelDataset::from_ticks`], except the grid conversion.
    pub fn from_kernel_ticks(ticks: Vec<KernelTick>, cfg: &KernelConfig) -> MMResult<Self> {
        let (tick, lot) = units(cfg)?;
        let max_price = tick
            .count(cfg.max_price)
            .and_then(|v| u64::try_from(v).ok())
            .ok_or_else(|| {
                MMError::InvalidConfiguration(format!(
                    "max_price {} is not a multiple of tick_size {}",
                    cfg.max_price, cfg.tick_size
                ))
            })?;
        let tail = cfg.latency.max_total_ns().ok_or_else(|| {
            MMError::InvalidConfiguration("total latency overflows u64 nanoseconds".to_string())
        })?;
        let mut previous: Option<u64> = None;
        for (row, t) in ticks.iter().enumerate() {
            if t.bid == 0 || t.ask == 0 {
                return Err(data_error(
                    row,
                    format!("non-positive price (bid {}, ask {})", t.bid, t.ask),
                ));
            }
            if t.bid >= t.ask {
                return Err(data_error(
                    row,
                    format!("bid {} >= ask {} ticks", t.bid, t.ask),
                ));
            }
            if t.ask > max_price {
                return Err(data_error(
                    row,
                    format!("ask {} above max_price {max_price} ticks", t.ask),
                ));
            }
            if let Some(prev) = previous
                && t.ts_ns < prev
            {
                return Err(data_error(
                    row,
                    format!("timestamp {} ns decreases from {prev} ns", t.ts_ns),
                ));
            }
            if t.ts_ns.checked_add(tail).is_none() {
                return Err(data_error(
                    row,
                    format!(
                        "timestamp {} ns plus the latency {tail} ns overflows",
                        t.ts_ns
                    ),
                ));
            }
            previous = Some(t.ts_ns);
        }
        let grid = DataGrid {
            tick_m: tick.mantissa,
            tick_s: tick.scale,
            lot_m: lot.mantissa,
            lot_s: lot.scale,
            max_price,
        };
        let len = u64::try_from(ticks.len())
            .map_err(|_| MMError::InvalidMarketState("dataset length overflows u64".to_string()))?;
        // Every integer goes straight into the hasher, little endian, in
        // the canonical order: grid, length, then each row.
        let mut hash = Fnv128::new();
        hash.update(&grid.tick_m.to_le_bytes());
        hash.update(&grid.tick_s.to_le_bytes());
        hash.update(&grid.lot_m.to_le_bytes());
        hash.update(&grid.lot_s.to_le_bytes());
        hash.update(&grid.max_price.to_le_bytes());
        hash.update(&len.to_le_bytes());
        for t in &ticks {
            for v in [t.ts_ns, t.bid, t.bid_lots, t.ask, t.ask_lots] {
                hash.update(&v.to_le_bytes());
            }
        }
        let id = DatasetId {
            scheme: DIGEST_FNV1A128,
            len,
            first_ts: ticks.first().map_or(0, |t| t.ts_ns),
            last_ts: ticks.last().map_or(0, |t| t.ts_ns),
            digest: hash.finish(),
            external: [0; 32],
        };
        Ok(Self { ticks, id, grid })
    }

    /// Sets the caller supplied digest in the identity.
    #[must_use]
    pub fn with_external_digest(mut self, external: [u8; 32]) -> Self {
        self.id.external = external;
        self
    }

    /// The rows.
    #[must_use]
    pub fn ticks(&self) -> &[KernelTick] {
        &self.ticks
    }

    /// Number of rows.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ticks.len()
    }

    /// Returns true if there are no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ticks.is_empty()
    }

    /// Identity of the dataset.
    #[must_use]
    pub fn id(&self) -> DatasetId {
        self.id
    }

    /// Row `index`, if any.
    pub(super) fn row(&self, index: u64) -> Option<&KernelTick> {
        self.ticks.get(usize::try_from(index).ok()?)
    }
}

fn units(cfg: &KernelConfig) -> MMResult<(Unit, Unit)> {
    let tick = Unit::new(cfg.tick_size).ok_or_else(|| {
        MMError::InvalidConfiguration(format!("tick_size must be positive, got {}", cfg.tick_size))
    })?;
    let lot = Unit::new(cfg.lot_size).ok_or_else(|| {
        MMError::InvalidConfiguration(format!("lot_size must be positive, got {}", cfg.lot_size))
    })?;
    Ok((tick, lot))
}
