//! Joint quotes for several correlated assets, under one capital budget.
//!
//! [`MultiAssetModel`] reproduces the asymptotic closed-form quotes of section
//! 3.3.3 of Bergault, Evangelista, Gueant and Vieira, *Closed-form
//! approximations in multi-asset market making*, arXiv:1810.04383v5: an
//! inventory in one asset moves the quotes of the others through the matrix
//! `Gamma`, so a hedged portfolio is skewed less than a concentrated one of the
//! same size. [`RiskView::Diagonal`] runs the same formula without the
//! correlations, for comparison.
//!
//! The model yields [`QuoteOffsets`], which are not orders. With the `margin`
//! feature too, `MultiAssetQuoter` turns them into quotes, each submitted to a
//! `margin-rs` risk gate, and reports the cause of each active constraint.
//!
//! This covers linear exposures only. There is no option logic here, and with
//! linear instruments the only Greek is the delta, which is the inventory.

mod matrix;
mod model;
#[cfg(feature = "margin")]
mod quoter;

pub use model::{
    AssetQuotingParams, MAX_ASSETS, MultiAssetModel, QuoteOffsets, QuotingObjective, RiskView,
};
#[cfg(feature = "margin")]
pub use quoter::{ActiveConstraint, AdmittedQuote, ConstraintKind, MultiAssetQuoter, QuoteRound};
