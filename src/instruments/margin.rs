//! Conversion of an [`InstrumentSpec`] into a `margin-rs` [`Instrument`].
//!
//! [`margin_instrument`] is a conversion only. It maps the fields exactly,
//! with no rounding, rescaling or currency substitution, and returns a typed
//! [`AdapterError`] for anything it cannot represent:
//!
//! | specification | `margin-rs` instrument | contract size |
//! |---|---|---|
//! | linear spot | [`SpotInstrument`]: base the underlying, quote the price asset | |
//! | linear future | [`LinearFuture`], settled in the settlement asset | multiplier: the size amount |
//! | linear perpetual | [`LinearPerpetual`] | multiplier: the size amount |
//! | inverse future | [`InverseFuture`], settled in the settlement asset (the underlying) | contract value: the size amount |
//! | inverse perpetual | [`InversePerpetual`] | contract value: the size amount |
//! | option | none: [`AdapterError::Unsupported`] | |
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
//! - option terms (options are not converted).
//!
//! The lot is not a risk minimum either. When the risk limits of an
//! instrument leave `min_quantity` unset, `margin-rs` uses one unit at
//! `contract_scale` (`10^-contract_scale`) as the minimum order size: with a
//! lot of `0.25` and a contract scale of `2`, that is `0.01`, not `0.25`.
//! Configure `min_quantity` as the lot to make the risk gate agree with the
//! venue.
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

use financial_types::instrument::{ContractKind, InstrumentSpec, Margining};
use margin_rs::{
    Asset, Instrument, InstrumentId, InverseFuture, InversePerpetual, InverseTerms, LinearFuture,
    LinearPerpetual, LinearTerms, MarginError, SpotInstrument,
};
use rust_decimal::Decimal;

/// The precision of a `margin-rs` instrument: the largest number of
/// fractional digits it accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MarginScales {
    /// Digits of execution prices and expiry settlement prices.
    pub price_scale: u32,
    /// Digits of contract quantities (the base quantity for spot).
    pub contract_scale: u32,
}

/// Why a specification has no `margin-rs` instrument.
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
    Margin(#[from] MarginError),
}

/// The number of fractional digits of `value`, trailing zeros ignored.
fn digits(value: Decimal) -> u32 {
    value.normalize().scale()
}

/// Checks that `given` holds the `required` digits of `grid`.
fn check_scale(
    field: &'static str,
    grid: &'static str,
    given: u32,
    required: u32,
) -> Result<(), AdapterError> {
    if given < required {
        return Err(AdapterError::ScaleBelowGrid {
            field,
            grid,
            given,
            required,
        });
    }
    Ok(())
}

/// Converts `spec` into a `margin-rs` instrument with the precision
/// `scales`; see the [module documentation](self) for the mapping.
///
/// # Errors
///
/// Checked in this order:
///
/// - [`AdapterError::Unsupported`] for an option;
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
    let (kind, margining) = (spec.kind(), spec.margining());
    if kind == ContractKind::Option {
        return Err(AdapterError::Unsupported { kind, margining });
    }
    let tick_digits = spec
        .price_ticks()
        .bands()
        .iter()
        .map(|band| digits(band.tick))
        .max()
        .unwrap_or(0);
    check_scale(
        "price_scale",
        "price ticks",
        scales.price_scale,
        tick_digits,
    )?;
    check_scale(
        "contract_scale",
        "lot",
        scales.contract_scale,
        digits(spec.lot()),
    )?;

    let MarginScales {
        price_scale,
        contract_scale,
    } = scales;
    let id = InstrumentId::new(spec.id().as_str())?;
    let settlement = Asset::new(spec.settlement_asset().as_str())?;
    let size = spec.size().amount();
    let linear = || {
        LinearTerms::new(
            id.clone(),
            settlement.clone(),
            size,
            price_scale,
            contract_scale,
        )
    };
    let inverse = || {
        InverseTerms::new(
            id.clone(),
            settlement.clone(),
            size,
            price_scale,
            contract_scale,
        )
    };
    let instrument = match (kind, margining, spec.expiry_ms()) {
        (ContractKind::Spot, Margining::Linear, None) => SpotInstrument::new(
            id.clone(),
            Asset::new(spec.underlying().as_str())?,
            Asset::new(spec.price_asset().as_str())?,
            price_scale,
            contract_scale,
        )?
        .into(),
        (ContractKind::Future, Margining::Linear, Some(expiry_ms)) => {
            LinearFuture::new(linear()?, expiry_ms).into()
        }
        (ContractKind::Perpetual, Margining::Linear, None) => {
            LinearPerpetual::new(linear()?).into()
        }
        (ContractKind::Future, Margining::Inverse, Some(expiry_ms)) => {
            InverseFuture::new(inverse()?, expiry_ms).into()
        }
        (ContractKind::Perpetual, Margining::Inverse, None) => {
            InversePerpetual::new(inverse()?).into()
        }
        _ => return Err(AdapterError::Unsupported { kind, margining }),
    };
    Ok(instrument)
}
