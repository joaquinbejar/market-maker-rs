//! Contract specifications shared with `margin-rs` and `arbitrage-rs`.
//!
//! Everything in [`financial_types::instrument`] is re-exported here: the
//! validated [`InstrumentSpec`](crate::instruments::InstrumentSpec), exact
//! money ([`Money`](crate::instruments::Money),
//! [`Ratio`](crate::instruments::Ratio),
//! [`MoneyRatio`](crate::instruments::MoneyRatio)) and the dated venue
//! snapshots in [`reference`](mod@crate::instruments::reference). The
//! formulas live in `financial_types` only, so every crate that prices a
//! contract computes the same exact amounts.
//!
//! [`orders`](crate::instruments::orders) checks orders and fills against a
//! specification, and [`futures`](crate::instruments::futures) follows a dated
//! future through its expiry and settlement, and
//! [`perpetual`](crate::instruments::perpetual) declares a perpetual's funding. With the `arbitrage` feature, `basis`
//! evaluates a spot and a dated future against each other, `calendar` two dated
//! futures under an assumed exit spread, and `roll` plans moving a position from one
//! dated future to the next. With the `margin` feature as well, the `margin` module
//! converts a specification into a `margin-rs` instrument and a fill into a
//! `margin-rs` execution.

pub use financial_types::instrument::*;

#[cfg(feature = "arbitrage")]
pub mod basis;
#[cfg(feature = "arbitrage")]
pub mod calendar;
pub mod futures;
#[cfg(feature = "margin")]
pub mod margin;
pub mod orders;
pub mod perpetual;
#[cfg(feature = "arbitrage")]
pub mod roll;
