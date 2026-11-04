//! Fee currency of an execution.

/// Asset in which a fill's fee is charged.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FeeAsset {
    /// The quote asset of the instrument (the default convention).
    #[default]
    Quote,
    /// The base asset of the instrument.
    Base,
    /// A third asset, named by its code (for example a venue token).
    Other(String),
    /// The settlement asset of a derivative contract.
    Settlement,
}
