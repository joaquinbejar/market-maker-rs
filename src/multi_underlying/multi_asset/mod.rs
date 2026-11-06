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
//! Those cover linear exposures, where the only Greek is the delta, which is the
//! inventory. With `options` and `margin`, `OptionQuoter` runs the same model over
//! option contracts, from their sensitivities to a declared factor covariance, and
//! holds each quote in a `margin-rs` ledger under one budget. Option values come
//! from the pricer, and none are computed here.

mod matrix;
mod model;
#[cfg(all(feature = "options", feature = "margin"))]
mod option_quoter;
#[cfg(feature = "margin")]
mod quoter;

pub use model::{
    AssetQuotingParams, MAX_ASSETS, MultiAssetModel, QuoteOffsets, QuotingObjective, RiskView,
};
#[cfg(all(feature = "options", feature = "margin"))]
pub use option_quoter::{
    AdmittedOptionQuote, ClippedEigenvalue, ContractTerms, FactorRisk, LimitedGreek, OptionCapital,
    OptionConstraint, OptionConstraintKind, OptionFill, OptionMarket, OptionQuoteRound,
    OptionQuoter, OptionQuoterConfig, RiskScenario, UnderlyingTerms,
};
#[cfg(feature = "margin")]
pub use quoter::{
    ActiveConstraint, AdmittedQuote, CapitalUse, ConstraintKind, MultiAssetQuoter, QuoteRound,
};
