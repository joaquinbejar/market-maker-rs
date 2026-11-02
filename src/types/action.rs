//! Trade direction and option style shared with the other crates.
//!
//! The public API uses [`Action`](crate::types::action::Action) (order
//! direction) and [`OptionStyle`](crate::types::action::OptionStyle) from
//! [`financial_types`], the same types margin-rs, arbitrage-rs and
//! OptionStratLib use, so no conversion glue is needed between them.
//!
//! `Action` is `#[non_exhaustive]` and has an `Other` variant. Only `Buy`
//! and `Sell` are supported: every method that takes or processes an order,
//! a fill or a side converts the action into the private `Dir` once, and
//! returns
//! [`MMError::UnsupportedAction`](crate::types::error::MMError::UnsupportedAction)
//! for any other before changing anything. The exceptions are documented
//! where they apply: [`PretradePolicy::check`](crate::risk::pretrade::PretradePolicy::check)
//! returns a rejection instead of an error, and a policy that receives a
//! fill with an unsupported side latches a fault on purpose.

use crate::types::error::{MMError, MMResult};

pub use financial_types::{Action, OptionStyle};

/// The supported directions, internally.
///
/// `Action` is a foreign `#[non_exhaustive]` enum with an `Other` variant, so
/// every input is converted here once, the only place that needs a wildcard
/// arm; all other matches are exhaustive on `Dir`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Dir {
    /// Buy (bid side).
    Buy,
    /// Sell (ask side).
    Sell,
}

impl TryFrom<Action> for Dir {
    type Error = MMError;

    #[inline]
    fn try_from(action: Action) -> MMResult<Self> {
        match action {
            Action::Buy => Ok(Self::Buy),
            Action::Sell => Ok(Self::Sell),
            // `Other`, or a variant added in a later financial_types.
            action => Err(MMError::UnsupportedAction { action }),
        }
    }
}

impl From<Dir> for Action {
    #[inline]
    fn from(dir: Dir) -> Self {
        match dir {
            Dir::Buy => Self::Buy,
            Dir::Sell => Self::Sell,
        }
    }
}

/// Serde adapters that keep the wire format of the event types (`"buy"`/
/// `"sell"`, `"call"`/`"put"`) after adopting the shared enums, whose own
/// serde representation is `"Buy"`/`"Call"`.
///
/// Each adapter goes through a private mirror of the former local enum, with
/// the same variants, order and `rename_all = "lowercase"`, so every serde
/// format (self-describing or not) sees exactly what it saw before. Only the
/// `events` feature (which enables `serde`) uses them.
#[cfg(feature = "events")]
pub(crate) mod serde_lowercase {
    /// `#[serde(with = ...)]` adapter for an [`Action`](super::Action)
    /// field: writes `"buy"`/`"sell"` and reads only those; any other
    /// action fails on both serialization and deserialization.
    pub(crate) mod action {
        use super::super::{Action, Dir};
        use serde::{Deserialize, Deserializer, Serialize, Serializer, ser};

        /// Mirror of the former `events::types::Side`.
        #[derive(Serialize, Deserialize)]
        #[serde(rename_all = "lowercase")]
        enum Wire {
            Buy,
            Sell,
        }

        /// Serializes a supported action as `"buy"` or `"sell"`.
        ///
        /// # Errors
        ///
        /// Fails for an action other than buy or sell.
        pub(crate) fn serialize<S: Serializer>(
            action: &Action,
            serializer: S,
        ) -> Result<S::Ok, S::Error> {
            let wire = match Dir::try_from(*action).map_err(ser::Error::custom)? {
                Dir::Buy => Wire::Buy,
                Dir::Sell => Wire::Sell,
            };
            wire.serialize(serializer)
        }

        /// Deserializes `"buy"` or `"sell"`.
        ///
        /// # Errors
        ///
        /// Fails for any other value.
        pub(crate) fn deserialize<'de, D: Deserializer<'de>>(
            deserializer: D,
        ) -> Result<Action, D::Error> {
            Ok(match Wire::deserialize(deserializer)? {
                Wire::Buy => Action::Buy,
                Wire::Sell => Action::Sell,
            })
        }
    }

    /// `#[serde(with = ...)]` adapter for an
    /// [`OptionStyle`](super::OptionStyle) field: writes and reads
    /// `"call"`/`"put"`.
    pub(crate) mod option_style {
        use super::super::OptionStyle;
        use serde::{Deserialize, Deserializer, Serialize, Serializer};

        /// Mirror of the former `events::types::OptionStyle`.
        #[derive(Serialize, Deserialize)]
        #[serde(rename_all = "lowercase")]
        enum Wire {
            Call,
            Put,
        }

        /// Serializes the style as `"call"` or `"put"`.
        ///
        /// # Errors
        ///
        /// Only the serializer's own errors.
        pub(crate) fn serialize<S: Serializer>(
            style: &OptionStyle,
            serializer: S,
        ) -> Result<S::Ok, S::Error> {
            let wire = match style {
                OptionStyle::Call => Wire::Call,
                OptionStyle::Put => Wire::Put,
            };
            wire.serialize(serializer)
        }

        /// Deserializes `"call"` or `"put"`.
        ///
        /// # Errors
        ///
        /// Fails for any other value.
        pub(crate) fn deserialize<'de, D: Deserializer<'de>>(
            deserializer: D,
        ) -> Result<OptionStyle, D::Error> {
            Ok(match Wire::deserialize(deserializer)? {
                Wire::Call => OptionStyle::Call,
                Wire::Put => OptionStyle::Put,
            })
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]
mod tests {
    use super::*;

    #[test]
    fn test_dir_from_buy_and_sell_round_trips() {
        for action in [Action::Buy, Action::Sell] {
            let dir = Dir::try_from(action).unwrap();
            assert_eq!(Action::from(dir), action);
        }
        assert_eq!(Dir::try_from(Action::Buy).unwrap(), Dir::Buy);
        assert_eq!(Dir::try_from(Action::Sell).unwrap(), Dir::Sell);
    }

    #[test]
    fn test_dir_from_other_returns_unsupported_action() {
        assert_eq!(
            Dir::try_from(Action::Other),
            Err(MMError::UnsupportedAction {
                action: Action::Other
            })
        );
    }

    #[test]
    fn test_action_serde_matches_former_connector_side() {
        // The former `execution::Side` derived serde with the default
        // representation; `Action` serializes the same way.
        assert_eq!(serde_json::to_string(&Action::Buy).unwrap(), r#""Buy""#);
        assert_eq!(serde_json::to_string(&Action::Sell).unwrap(), r#""Sell""#);
        assert_eq!(
            serde_json::from_str::<Action>(r#""Buy""#).unwrap(),
            Action::Buy
        );
        assert_eq!(
            serde_json::from_str::<Action>(r#""Sell""#).unwrap(),
            Action::Sell
        );
    }

    #[test]
    fn test_action_display_matches_former_connector_side() {
        assert_eq!(Action::Buy.to_string(), "Buy");
        assert_eq!(Action::Sell.to_string(), "Sell");
    }
}
