//! Orders and fills against a contract specification.
//!
//! Orders and fills name their instrument by symbol and do not carry its
//! specification. These functions take the specification from the caller, check
//! that it is the one named, and refuse what does not fit it, with a typed error;
//! nothing is rounded or converted.
//!
//! # Units
//!
//! Here `OrderRequest::quantity` and `Fill::quantity` are in contracts of the
//! specification. On a contract sized one unit of the underlying, that is also
//! the base quantity; on any other, for example an inverse contract sized in
//! value, it is not. [`Fill::notional`] is `price * quantity`, which is correct
//! only for a linear contract of multiplier one: [`fill_notional`] gives the
//! notional of the contract, in its asset.
//!
//! # Orders and fills
//!
//! [`check_order`] applies the venue's grid to an order before it is sent. It
//! has no clock, so it does not check an expiry. It also cannot tell base
//! currency from contracts: with a multiplier other than one, a size in base
//! passes when it happens to fall on the lot.
//!
//! [`fill_notional`] does not apply the grid to a fill: a fill already happened,
//! and the grid comes from a dated snapshot that may be stale. It does check the
//! fill is of this instrument, by symbol: that is its identity, not the version
//! of the terms it traded under.

use financial_types::instrument::{ContractKind, InstrumentError, InstrumentSpec, MoneyRatio};

use crate::Decimal;
use crate::execution::{Fill, OrderRequest, OrderType};
use crate::types::action::Dir;
use financial_types::Action;

/// Why an order or a fill does not fit a specification.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum OrderSpecError {
    /// The order or the fill is of another instrument.
    #[error("symbol {symbol} checked against instrument {instrument}")]
    SymbolMismatch {
        /// The symbol of the order or the fill.
        symbol: String,
        /// The specification's identifier.
        instrument: String,
    },
    /// The order's side is neither buy nor sell.
    #[error("unsupported order side {side}: only buy and sell are supported")]
    UnsupportedSide {
        /// The side given.
        side: Action,
    },
    /// A limit or post-only order has no price.
    #[error("{order_type:?} order without a price")]
    MissingPrice {
        /// The order type.
        order_type: OrderType,
    },
    /// A market order has a price, which would not be applied.
    #[error("market order with a price of {price}")]
    UnexpectedPrice {
        /// The price given.
        price: Decimal,
    },
    /// The price of an option fill is a premium, so it has no notional here;
    /// `InstrumentSpec::option_premium` gives what it costs.
    #[error("{instrument} is an option: its fill price is a premium")]
    OptionNotional {
        /// The specification's identifier.
        instrument: String,
    },
    /// The specification refused a price or a size.
    #[error(transparent)]
    Spec(#[from] InstrumentError),
}

/// Checks an order against `spec` before it is sent.
///
/// # Errors
///
/// Checked in this order:
///
/// - [`OrderSpecError::SymbolMismatch`] unless the symbol is the identifier of
///   `spec`, compared exactly;
/// - [`OrderSpecError::UnsupportedSide`] for a side other than buy or sell;
/// - [`OrderSpecError::MissingPrice`] for a limit or post-only order without a
///   price, and [`OrderSpecError::UnexpectedPrice`] for a market order with one;
/// - [`OrderSpecError::Spec`] for a price off its tick or a quantity off the
///   lot, or one that is not positive.
pub fn check_order(spec: &InstrumentSpec, order: &OrderRequest) -> Result<(), OrderSpecError> {
    if order.symbol != spec.id().as_str() {
        return Err(OrderSpecError::SymbolMismatch {
            symbol: order.symbol.clone(),
            instrument: spec.id().as_str().to_string(),
        });
    }
    if Dir::try_from(order.side).is_err() {
        return Err(OrderSpecError::UnsupportedSide { side: order.side });
    }
    match (order.order_type, order.price) {
        (OrderType::Limit | OrderType::PostOnly, Some(price)) => spec.check_price(price)?,
        (OrderType::Limit | OrderType::PostOnly, None) => {
            return Err(OrderSpecError::MissingPrice {
                order_type: order.order_type,
            });
        }
        (OrderType::Market, Some(price)) => {
            return Err(OrderSpecError::UnexpectedPrice { price });
        }
        (OrderType::Market, None) => {}
    }
    spec.check_lot(order.quantity)?;
    Ok(())
}

/// The notional of `fill` under `spec`, in its asset, exactly: `c·m·p` in the
/// price asset for a contract sized in the underlying, `c·CV/p` in the
/// underlying for an inverse contract sized in value.
///
/// # Errors
///
/// [`OrderSpecError::SymbolMismatch`] unless the fill's symbol is the identifier
/// of `spec`, compared exactly; [`OrderSpecError::OptionNotional`] for an option;
/// and [`OrderSpecError::Spec`] when `InstrumentSpec::notional` refuses it: a price
/// or a quantity that is not positive, or a product that is not exact.
pub fn fill_notional(spec: &InstrumentSpec, fill: &Fill) -> Result<MoneyRatio, OrderSpecError> {
    if fill.symbol != spec.id().as_str() {
        return Err(OrderSpecError::SymbolMismatch {
            symbol: fill.symbol.clone(),
            instrument: spec.id().as_str().to_string(),
        });
    }
    if spec.kind() == ContractKind::Option {
        return Err(OrderSpecError::OptionNotional {
            instrument: spec.id().as_str().to_string(),
        });
    }
    Ok(spec.notional(fill.quantity, fill.price)?)
}
