//! Conversion of an [`InstrumentSpec`] into a `margin-rs` [`Instrument`], and of
//! a fill of it into a `margin-rs` execution ([`fill_execution`]).
//!
//! [`margin_instrument`] delegates to `margin_rs::Instrument::from_spec`, so
//! the mapping has one implementation, in `margin-rs`, and this module keeps
//! the public types of this crate.
//!
//! Both are conversions only. They map the fields exactly, with no rounding,
//! rescaling or currency substitution, and return a typed [`AdapterError`]
//! for anything they cannot represent. The instruments:
//!
//! | specification | `margin-rs` instrument | contract size |
//! |---|---|---|
//! | linear spot | [`SpotInstrument`](margin_rs::SpotInstrument): base the underlying, quote the price asset | |
//! | linear future | [`LinearFuture`](margin_rs::LinearFuture), settled in the settlement asset | multiplier: the size amount |
//! | linear perpetual | [`LinearPerpetual`](margin_rs::LinearPerpetual) | multiplier: the size amount |
//! | inverse future | [`InverseFuture`](margin_rs::InverseFuture), settled in the settlement asset (the underlying) | contract value: the size amount |
//! | inverse perpetual | [`InversePerpetual`](margin_rs::InversePerpetual) | contract value: the size amount |
//! | option | [`OptionInstrument`](margin_rs::instrument::OptionInstrument), as `margin-rs` 0.2 maps it | multiplier: the size amount |
//!
//! The identifier and the asset codes go through the `margin-rs`
//! constructors, and the expiry (milliseconds since the epoch) is passed
//! unchanged.
//!
//! # Scales
//!
//! `margin-rs` checks the number of fractional digits of execution prices
//! and of expiry settlement prices against `price_scale`, and of execution
//! quantities against `contract_scale` (`quantity_scale` for spot). Venues
//! often publish marks, index and delivery prices with more digits than the
//! order tick, so the scales are explicit arguments ([`MarginScales`]),
//! never inferred. Each must be at least the number of digits of the price
//! ticks and of the lot, trailing zeros ignored, so that every price and
//! size on the venue grid is accepted ([`AdapterError::ScaleBelowGrid`]).
//!
//! The scales only cap the digits: they do not enforce the grid. A price of
//! `0.3` has one digit but is off a `0.5` tick; check orders with
//! [`InstrumentSpec::check_price`] and [`InstrumentSpec::check_lot`].
//!
//! # What the conversion does not carry
//!
//! The conversion is not reversible. A `margin-rs` instrument has no field
//! for:
//!
//! - the tick schedule and the lot (see above);
//! - the underlying of a linear derivative: its multiplier is in units of
//!   the underlying, and its prices and settlement are in the price asset;
//! - the price asset of an inverse derivative: its contract value and
//!   prices are in that currency, which is not a ledger asset;
//! - for an option, whatever `margin-rs` leaves out of its option terms.
//!
//! The lot is not a risk minimum either. When the risk limits of an
//! instrument leave `min_quantity` unset, `margin-rs` uses one unit at
//! `contract_scale` (`10^-contract_scale`) as the minimum order size: with a
//! lot of `0.25` and a contract scale of `2`, that is `0.01`, not `0.25`.
//! Configure `min_quantity` as the lot to make the risk gate agree with the
//! venue.
//!
//! # Which `margin-rs`
//!
//! This module needs a `margin-rs` with its `instruments` feature, 0.1.1 or
//! later. The dependency stays `"0.1"`: with `margin` and `instruments` both
//! on, Cargo refuses a version without that feature when it resolves, naming
//! the missing feature, and `margin` alone still accepts 0.1.0. A lockfile
//! already at 0.1.0 is not upgraded by itself: adding `instruments` fails to
//! resolve until `cargo update -p margin-rs` is run.
//!
//! # Errors from `margin-rs`
//!
//! `From<MarginError>` for [`AdapterError`] is the only mapping, so a caller's
//! own `?` on `Instrument::from_spec` gives the same typed variants as
//! [`margin_instrument`]. The identifier those two `margin-rs` errors carry
//! is dropped, because the public variants have no field for it.
//!
//! # Checks left to the ledger
//!
//! Some checks need the scale of the settlement asset, which a
//! specification does not hold; `margin-rs` makes them when the instrument
//! is registered (`Ledger::register_instrument`), as a `MarginError`:
//!
//! - every asset of the instrument must be registered;
//! - the quantity scale of a spot instrument must not exceed the scale of
//!   its base asset;
//! - the scale of the settlement asset of an inverse instrument plus its
//!   price scale must not exceed 28.

use financial_types::Action;
use financial_types::instrument::{ContractKind, InstrumentSpec, Margining};
use margin_rs::instrument::InstrumentScales;
use margin_rs::ledger::Execution;
use margin_rs::{Asset, ExecutionId, Instrument, InstrumentId, MarginError};

use crate::Decimal;
use crate::execution::Fill;
use crate::types::action::Dir;

/// The precision of a `margin-rs` instrument: the largest number of
/// fractional digits it accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MarginScales {
    /// Digits of execution prices and expiry settlement prices.
    pub price_scale: u32,
    /// Digits of contract quantities (the base quantity for spot).
    pub contract_scale: u32,
}

/// Why a specification has no `margin-rs` instrument, or a fill no execution.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum AdapterError {
    /// `margin-rs` has no instrument for this kind of contract.
    #[error("margin-rs has no {margining:?} {kind:?} instrument")]
    Unsupported {
        /// The kind of contract.
        kind: ContractKind,
        /// Its margining.
        margining: Margining,
    },
    /// A scale has fewer digits than the venue grid.
    #[error("{field} {given} is below the {required} digits of the {grid}")]
    ScaleBelowGrid {
        /// `"price_scale"` or `"contract_scale"`.
        field: &'static str,
        /// What the grid is: `"price ticks"` or `"lot"`.
        grid: &'static str,
        /// The scale given.
        given: u32,
        /// The digits of the grid, trailing zeros ignored.
        required: u32,
    },
    /// A `margin-rs` constructor refused a field.
    #[error(transparent)]
    Margin(MarginError),
    /// A fill is of another instrument.
    #[error("fill of {symbol} converted with the specification of {instrument}")]
    SymbolMismatch {
        /// The fill's symbol.
        symbol: String,
        /// The specification's identifier.
        instrument: String,
    },
    /// A fill's side is neither buy nor sell.
    #[error("unsupported fill side {side}: only buy and sell are supported")]
    UnsupportedSide {
        /// The side given.
        side: Action,
    },
    /// A fill has no trade identifier, so it could not be told from another.
    #[error("fill without a trade id")]
    EmptyTradeId,
    /// A fill's price or quantity is not positive.
    #[error("fill {field} must be positive, got {value}")]
    NotPositive {
        /// `"price"` or `"quantity"`.
        field: &'static str,
        /// The value given.
        value: Decimal,
    },
}

/// The one mapping from a `margin-rs` error: its two specification errors
/// become the typed variants above, without their instrument identifier, and
/// everything else is [`AdapterError::Margin`].
impl From<MarginError> for AdapterError {
    fn from(error: MarginError) -> Self {
        match error {
            MarginError::UnsupportedSpec {
                kind, margining, ..
            } => Self::Unsupported { kind, margining },
            MarginError::ScaleBelowGrid {
                field,
                grid,
                given,
                required,
                ..
            } => Self::ScaleBelowGrid {
                field,
                grid,
                given,
                required,
            },
            // `MarginError` is `#[non_exhaustive]` in `margin-rs`, so this
            // arm is required; a variant added there stays `Margin`.
            other => Self::Margin(other),
        }
    }
}

/// Converts `spec` into a `margin-rs` instrument with the precision
/// `scales`; see the [module documentation](self) for the mapping.
///
/// # Errors
///
/// Checked in this order:
///
/// - [`AdapterError::Unsupported`] for a specification `margin-rs` does not convert;
/// - [`AdapterError::ScaleBelowGrid`] if `scales.price_scale` is below the
///   digits of a price tick, or `scales.contract_scale` below the digits of
///   the lot;
/// - [`AdapterError::Margin`] if a `margin-rs` constructor refuses a field,
///   such as a scale above its maximum, or a contract value whose digits
///   plus `contract_scale` exceed 28.
pub fn margin_instrument(
    spec: &InstrumentSpec,
    scales: MarginScales,
) -> Result<Instrument, AdapterError> {
    let scales = InstrumentScales {
        price_scale: scales.price_scale,
        contract_scale: scales.contract_scale,
    };
    Ok(Instrument::from_spec(spec, scales)?)
}

/// The `margin-rs` execution of a venue `fill` of `spec`, to book in a ledger.
///
/// A ledger keys executions by identifier alone, and a venue may number its
/// trades per symbol, so the identifier is
/// `<length of the instrument id>:<instrument id>:<trade_id>`: stable, so booking
/// the same fill twice counts it once, and with no two pairs giving the same
/// text. It is unique for one venue and one account, which is what one ledger
/// holds; a ledger mixing venues would need the venue in the instrument id.
///
/// The quantity is in contracts of `spec` and the time is the fill's. A zero fee
/// is no fee, whatever `fee_currency` says; any other, a negative one being a
/// rebate, is in `fee_currency`, and the ledger refuses an asset it has not
/// registered, so a fee is never lost.
///
/// A fill already happened, so the tick and the lot of `spec`, which come from
/// a dated snapshot, are not applied to it. A real fill can still be refused,
/// and these errors must not be dropped, since the fill would be lost: here for
/// example for a `fee_currency` that is not a valid asset code, which is written
/// exactly (`usdt` is not `USDT`), and by the ledger for example for digits above
/// the instrument's scales, an unregistered asset, a time at or after a future's
/// expiry, or, through `Ledger::book_execution`, a balance that does not cover it
/// (`Ledger::book_execution_allowing_overdraft` books it anyway).
///
/// # Errors
///
/// Checked in this order:
///
/// - [`AdapterError::SymbolMismatch`] unless the fill's symbol is the identifier
///   of `spec`, compared exactly;
/// - [`AdapterError::Unsupported`] for an option, whose fill needs its premium and its time
///   checked against the expiry, which this conversion does not do;
/// - [`AdapterError::UnsupportedSide`] for a side other than buy or sell;
/// - [`AdapterError::EmptyTradeId`] for an empty `trade_id`;
/// - [`AdapterError::NotPositive`] for a price or a quantity that is not
///   positive;
/// - [`AdapterError::Margin`] if `margin-rs` refuses the identifiers or the fee
///   asset as written: an execution identifier above 64 bytes or with
///   whitespace, never shortened, or an asset code that is not valid.
pub fn fill_execution(spec: &InstrumentSpec, fill: &Fill) -> Result<Execution, AdapterError> {
    if fill.symbol != spec.id().as_str() {
        return Err(AdapterError::SymbolMismatch {
            symbol: fill.symbol.clone(),
            instrument: spec.id().as_str().to_string(),
        });
    }
    if spec.kind() == ContractKind::Option {
        return Err(AdapterError::Unsupported {
            kind: spec.kind(),
            margining: spec.margining(),
        });
    }
    if Dir::try_from(fill.side).is_err() {
        return Err(AdapterError::UnsupportedSide { side: fill.side });
    }
    if fill.trade_id.is_empty() {
        return Err(AdapterError::EmptyTradeId);
    }
    for (field, value) in [("price", fill.price), ("quantity", fill.quantity)] {
        if value <= Decimal::ZERO {
            return Err(AdapterError::NotPositive { field, value });
        }
    }
    let instrument = spec.id().as_str();
    let execution = Execution::new(
        execution_id(instrument, &fill.trade_id)?,
        InstrumentId::new(instrument)?,
        fill.side,
        fill.price,
        fill.quantity,
    )
    .with_executed_at(fill.timestamp);
    Ok(if fill.fee.is_zero() {
        execution
    } else {
        execution.with_fee(Asset::new(fill.fee_currency.as_str())?, fill.fee)
    })
}

/// The ledger's identifier of a trade on a contract: a venue numbers its trades per
/// contract, so the contract is part of it.
pub(crate) fn execution_id(instrument: &str, trade_id: &str) -> Result<ExecutionId, MarginError> {
    ExecutionId::new(format!("{}:{}:{}", instrument.len(), instrument, trade_id))
}

/// The scales of a registered derivative.
pub(crate) fn registered_scales(instrument: &Instrument) -> Option<InstrumentScales> {
    let (price_scale, contract_scale) = match instrument.linear_terms() {
        Some(terms) => (terms.price_scale(), terms.contract_scale()),
        None => {
            let terms = instrument.inverse_terms()?;
            (terms.price_scale(), terms.contract_scale())
        }
    };
    Some(InstrumentScales {
        price_scale,
        contract_scale,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use financial_types::OptionStyle;
    use financial_types::instrument::InstrumentId as SpecId;
    use financial_types::instrument::reference::{binance_usdm, deribit};
    use margin_rs::InstrumentId;
    use rust_decimal_macros::dec;

    fn scales(price_scale: u32, contract_scale: u32) -> InstrumentScales {
        InstrumentScales {
            price_scale,
            contract_scale,
        }
    }

    #[test]
    fn test_from_margin_error_gives_the_typed_variants() {
        assert_eq!(
            AdapterError::from(MarginError::UnsupportedSpec {
                instrument: InstrumentId::new("X").unwrap(),
                kind: ContractKind::Option,
                margining: Margining::Inverse,
            }),
            AdapterError::Unsupported {
                kind: ContractKind::Option,
                margining: Margining::Inverse,
            }
        );
        assert_eq!(
            AdapterError::from(MarginError::ScaleBelowGrid {
                instrument: InstrumentId::new("X").unwrap(),
                field: "price_scale",
                grid: "price ticks",
                given: 0,
                required: 1,
            }),
            AdapterError::ScaleBelowGrid {
                field: "price_scale",
                grid: "price ticks",
                given: 0,
                required: 1,
            }
        );
    }

    #[test]
    fn test_from_margin_error_keeps_any_other_error_as_margin() {
        let error = MarginError::invalid_input("price_scale", "scale must be at most 18");
        assert_eq!(
            AdapterError::from(error.clone()),
            AdapterError::Margin(error)
        );
    }

    /// A caller's own `?` on `Instrument::from_spec` takes the same path as
    /// `margin_instrument`.
    #[test]
    fn test_from_spec_errors_convert_as_margin_instrument_reports_them() {
        let option = deribit::btc_option(
            SpecId::new("BTC-27NOV26-100000-C").unwrap(),
            1_795_766_400_000,
            dec!(100000),
            OptionStyle::Call,
        )
        .unwrap();
        let binance = binance_usdm::btcusdt_perpetual().unwrap();
        // margin-rs 0.2 converts options, so an option is no longer an error here.
        let converted = margin_instrument(
            &option,
            MarginScales {
                price_scale: 4,
                contract_scale: 1,
            },
        )
        .unwrap();
        assert!(converted.option_terms().is_some());
        assert_eq!(
            Ok(converted),
            Instrument::from_spec(&option, scales(4, 1)).map_err(AdapterError::from)
        );
        for (spec, (price_scale, contract_scale)) in [(&binance, (0, 3)), (&binance, (19, 3))] {
            let direct = Instrument::from_spec(spec, scales(price_scale, contract_scale))
                .map_err(AdapterError::from);
            let adapter = margin_instrument(
                spec,
                MarginScales {
                    price_scale,
                    contract_scale,
                },
            );
            assert!(adapter.is_err());
            assert_eq!(adapter, direct);
        }
    }
}
