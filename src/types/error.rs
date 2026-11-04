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
    /// [`Action::Other`]. It is returned before any state changes, except
    /// that a pretrade policy receiving such a fill also latches a fault.
    #[error("unsupported action {action}: only buy and sell are supported")]
    UnsupportedAction {
        /// The action given.
        action: Action,
    } = 7,

    /// A lock guarding shared state was poisoned.
    ///
    /// This error occurs when another thread panicked while holding the
    /// lock, so the protected state may be inconsistent.
    #[error("lock poisoned: {0}")]
    LockPoisoned(String) = 8,

    /// The contract cannot be valued by the model asked for.
    ///
    /// This error occurs when an option has an exercise style or a payoff
    /// the pricer does not support (for example an American or an exotic
    /// option given to the European Black-Scholes-Merton pricer). It is
    /// never replaced by an approximate value.
    #[error("unsupported contract: {reason}")]
    UnsupportedContract {
        /// Why the contract is not supported.
        reason: String,
    } = 9,

    /// A price whose true value is positive is below the smallest positive
    /// `f64` (about 4.9e-324), so it cannot be represented.
    ///
    /// Returned by `options::pricing::black_undiscounted` (feature `options`)
    /// for `w > 0`, where the Black price is strictly positive. It is never
    /// replaced by 0; an implied volatility solver can treat it as "below
    /// the representable range" and must propagate every other error. It is
    /// a numerical limit: [`MMError::is_numerical_error`] is true for it.
    #[error("numerical error: {}", PRICE_UNDERFLOW_MESSAGE)]
    PriceUnderflow = 10,

    /// A fill was not booked: this call left the Greeks unchanged, so they
    /// miss it.
    ///
    /// The risk manager stays fail closed until an explicit reset or
    /// reconciliation. Booking the fill again cannot count this call twice,
    /// because it applied nothing. That is all it promises: the next attempt
    /// may fail too, and one that succeeds does not recover the manager,
    /// because nothing proves which fill it was. The message carries the cause.
    #[error("fill not booked, the Greeks miss it: {0}")]
    FillNotBooked(String) = 11,

    /// A fill was booked, but the risk check after it failed.
    ///
    /// The Greeks include the fill: booking it again counts it twice. The
    /// message carries the cause.
    #[error("fill booked, but the risk check after it failed: {0}")]
    FillBookedCheckFailed(String) = 12,

    /// An answer was asked from Greeks known to miss a fill that could not be
    /// booked ([`MMError::FillNotBooked`]).
    ///
    /// It lasts until the risk manager is explicitly reset or reconciled.
    #[error("{}", GREEKS_MISS_FILL_MESSAGE)]
    GreeksMissFill = 13,

    /// The chain risk status does not allow quoting, so no quote was produced.
    ///
    /// Any quote resting from an earlier refresh should be withdrawn. Retrying
    /// does not help until the status changes, and none of the `is_*` helpers
    /// is true for it. The message carries the status.
    #[error("quoting halted: {0}")]
    QuotingHalted(String) = 14,
}

/// The message of [`MMError::PriceUnderflow`].
const PRICE_UNDERFLOW_MESSAGE: &str = "Black price is positive but below the smallest positive f64";

/// The message of [`MMError::GreeksMissFill`].
const GREEKS_MISS_FILL_MESSAGE: &str = "the Greeks miss a fill that could not be booked";

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

    /// Returns true if this error is related to numerical issues:
    /// [`MMError::NumericalError`] and [`MMError::PriceUnderflow`].
    #[must_use]
    pub fn is_numerical_error(&self) -> bool {
        matches!(self, Self::NumericalError(_) | Self::PriceUnderflow)
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
            | Self::ConnectionError(msg)
            | Self::LockPoisoned(msg)
            | Self::FillNotBooked(msg)
            | Self::FillBookedCheckFailed(msg)
            | Self::QuotingHalted(msg) => msg,
            Self::UnsupportedContract { reason } => reason,
            Self::UnsupportedAction { .. } => "only buy and sell are supported",
            Self::PriceUnderflow => PRICE_UNDERFLOW_MESSAGE,
            Self::GreeksMissFill => GREEKS_MISS_FILL_MESSAGE,
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

        let halted = MMError::QuotingHalted("Breached".to_string());
        assert!(!halted.is_configuration_error());
        assert!(!halted.is_market_state_error());
        assert!(!halted.is_numerical_error());
        assert!(!halted.is_connection_error());
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
            MMError::FillNotBooked("test".to_string()),
            MMError::FillBookedCheckFailed("test".to_string()),
            MMError::GreeksMissFill,
            MMError::QuotingHalted("test".to_string()),
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

    #[test]
    fn test_price_underflow_display_message_and_class() {
        let err = MMError::PriceUnderflow;
        assert_eq!(
            err.to_string(),
            "numerical error: Black price is positive but below the smallest positive f64"
        );
        assert_eq!(
            err.message(),
            "Black price is positive but below the smallest positive f64"
        );
        assert!(err.is_numerical_error());
        assert!(!err.is_configuration_error());
        assert!(!err.is_market_state_error());
        assert!(!err.is_connection_error());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_price_underflow_serialization() {
        let err = MMError::PriceUnderflow;
        let json = serde_json::to_string(&err).unwrap();
        assert_eq!(json, r#"{"type":"PriceUnderflow"}"#);
        let deserialized: MMError = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, err);
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
