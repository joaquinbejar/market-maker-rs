//! Error types for the market making library.

use thiserror::Error;

use crate::types::action::Action;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Result type alias for market making operations.
pub type MMResult<T> = std::result::Result<T, MMError>;

/// Main error type for the market making library.
///
/// This enum represents all possible errors that can occur during market making operations.
/// It uses tagged serialization for clear error identification in serialized formats.
#[derive(Error, Debug, Clone, PartialEq)]
#[repr(u8)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(tag = "type", content = "details"))]
pub enum MMError {
    /// Invalid configuration parameter.
    ///
    /// This error occurs when a strategy configuration has invalid parameters,
    /// such as negative risk aversion or invalid time values.
    #[error("invalid configuration: {0}")]
    InvalidConfiguration(String) = 0,

    /// Invalid market state.
    ///
    /// This error occurs when market data is invalid, such as negative prices,
    /// negative volatility, or NaN values.
    #[error("invalid market state: {0}")]
    InvalidMarketState(String) = 1,

    /// Numerical error (overflow, NaN, infinity, etc.).
    ///
    /// This error occurs when numerical calculations produce invalid results,
    /// such as division by zero, overflow, or floating-point errors.
    #[error("numerical error: {0}")]
    NumericalError(String) = 2,

    /// Invalid position update.
    ///
    /// This error occurs when a position update is invalid, such as attempting
    /// to close more than the current position or invalid fill prices.
    #[error("invalid position update: {0}")]
    InvalidPositionUpdate(String) = 3,

    /// Invalid quote generation.
    ///
    /// This error occurs when quote generation fails, such as when bid >= ask,
    /// negative spreads, or quotes violate minimum spread constraints.
    #[error("invalid quote generation: {0}")]
    InvalidQuoteGeneration(String) = 4,

    /// Invalid timestamp.
    ///
    /// This error occurs when timestamps are invalid, such as current time
    /// being after terminal time, or time moving backwards.
    #[error("invalid timestamp: {0}")]
    InvalidTimestamp(String) = 5,

    /// Connection error.
    ///
    /// This error occurs when there are issues with network connectivity,
    /// such as failed connections, timeouts, or disconnections.
    #[error("connection error: {0}")]
    ConnectionError(String) = 6,

    /// Unsupported trade direction.
    ///
    /// This error occurs when an order, fill or query carries an
    /// [`Action`] other than [`Action::Buy`] or [`Action::Sell`], for example
    /// [`Action::Other`]. It is returned before any state changes.
    #[error("unsupported action {action}: only buy and sell are supported")]
    UnsupportedAction {
        /// The action given.
        action: Action,
    } = 7,
}

impl MMError {
    /// Returns true if this error is related to configuration issues.
    #[must_use]
    pub fn is_configuration_error(&self) -> bool {
        matches!(self, Self::InvalidConfiguration(_))
    }

    /// Returns true if this error is related to market state issues.
    #[must_use]
    pub fn is_market_state_error(&self) -> bool {
        matches!(self, Self::InvalidMarketState(_))
    }

    /// Returns true if this error is related to numerical issues.
    #[must_use]
    pub fn is_numerical_error(&self) -> bool {
        matches!(self, Self::NumericalError(_))
    }

    /// Returns true if this error is related to connection issues.
    #[must_use]
    pub fn is_connection_error(&self) -> bool {
        matches!(self, Self::ConnectionError(_))
    }

    /// Returns the error message as a string slice.
    #[must_use]
    pub fn message(&self) -> &str {
        match self {
            Self::InvalidConfiguration(msg)
            | Self::InvalidMarketState(msg)
            | Self::NumericalError(msg)
            | Self::InvalidPositionUpdate(msg)
            | Self::InvalidQuoteGeneration(msg)
            | Self::InvalidTimestamp(msg)
            | Self::ConnectionError(msg) => msg,
            Self::UnsupportedAction { .. } => "only buy and sell are supported",
        }
    }
}

/// Legacy type alias for backward compatibility.
///
/// Prefer using [`MMResult`] for new code.
#[deprecated(since = "0.1.0", note = "use MMResult instead")]
pub type Result<T> = MMResult<T>;

#[cfg(test)]
mod tests {
    use super::super::error::{MMError, MMResult};
    use crate::types::action::Action;

    #[test]
    fn test_error_creation() {
        let err = MMError::InvalidConfiguration("test error".to_string());
        assert_eq!(err.to_string(), "invalid configuration: test error");
    }

    #[test]
    fn test_error_type_checking() {
        let config_err = MMError::InvalidConfiguration("bad config".to_string());
        assert!(config_err.is_configuration_error());
        assert!(!config_err.is_market_state_error());
        assert!(!config_err.is_numerical_error());

        let market_err = MMError::InvalidMarketState("bad market".to_string());
        assert!(!market_err.is_configuration_error());
        assert!(market_err.is_market_state_error());
        assert!(!market_err.is_numerical_error());

        let num_err = MMError::NumericalError("overflow".to_string());
        assert!(!num_err.is_configuration_error());
        assert!(!num_err.is_market_state_error());
        assert!(num_err.is_numerical_error());
    }

    #[test]
    fn test_error_message() {
        let err = MMError::InvalidConfiguration("test message".to_string());
        assert_eq!(err.message(), "test message");

        let err2 = MMError::NumericalError("overflow detected".to_string());
        assert_eq!(err2.message(), "overflow detected");

        let err3 = MMError::InvalidTimestamp("time error".to_string());
        assert_eq!(err3.message(), "time error");

        let err4 = MMError::InvalidMarketState("bad market".to_string());
        assert_eq!(err4.message(), "bad market");

        let err5 = MMError::InvalidPositionUpdate("bad position".to_string());
        assert_eq!(err5.message(), "bad position");

        let err6 = MMError::InvalidQuoteGeneration("bad quote".to_string());
        assert_eq!(err6.message(), "bad quote");
    }

    #[test]
    fn test_result_type() {
        // Test successful result
        fn get_ok_result() -> MMResult<i32> {
            Ok(42)
        }
        let ok_result = get_ok_result();
        assert!(ok_result.is_ok());
        assert_eq!(ok_result.unwrap(), 42);

        // Test error result
        fn get_err_result() -> MMResult<i32> {
            Err(MMError::NumericalError("overflow".to_string()))
        }
        let err_result = get_err_result();
        assert!(err_result.is_err());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_error_serialization() {
        use serde_json;

        let err = MMError::InvalidConfiguration("negative gamma".to_string());
        let json = serde_json::to_string(&err).unwrap();

        // Verify tagged serialization
        assert!(json.contains(r#""type":"InvalidConfiguration"#));
        assert!(json.contains(r#""details":"negative gamma"#));

        // Verify deserialization
        let deserialized: MMError = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, err);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_all_error_variants_serialization() {
        use serde_json;

        let errors = vec![
            MMError::InvalidConfiguration("test".to_string()),
            MMError::InvalidMarketState("test".to_string()),
            MMError::NumericalError("test".to_string()),
            MMError::InvalidPositionUpdate("test".to_string()),
            MMError::InvalidQuoteGeneration("test".to_string()),
            MMError::InvalidTimestamp("test".to_string()),
        ];

        for err in errors {
            let json = serde_json::to_string(&err).unwrap();
            let deserialized: MMError = serde_json::from_str(&json).unwrap();
            assert_eq!(deserialized, err);
        }
    }

    #[test]
    fn test_unsupported_action_display_and_message() {
        let err = MMError::UnsupportedAction {
            action: Action::Other,
        };
        assert_eq!(
            err.to_string(),
            "unsupported action Other: only buy and sell are supported"
        );
        assert_eq!(err.message(), "only buy and sell are supported");
        assert!(!err.is_configuration_error());
        assert!(!err.is_market_state_error());
        assert!(!err.is_numerical_error());
        assert!(!err.is_connection_error());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_unsupported_action_serialization() {
        let err = MMError::UnsupportedAction {
            action: Action::Other,
        };
        let json = serde_json::to_string(&err).unwrap();
        assert_eq!(
            json,
            r#"{"type":"UnsupportedAction","details":{"action":"Other"}}"#
        );
        let deserialized: MMError = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, err);
    }
}
