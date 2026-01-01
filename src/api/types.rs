//! API types and data structures.

use crate::Decimal;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// API configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConfig {
    /// Bind address for the API server.
    pub bind_address: String,
    /// Enable CORS for frontend access.
    pub enable_cors: bool,
    /// Allowed origins for CORS.
    pub cors_origins: Vec<String>,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            bind_address: "0.0.0.0:8080".to_string(),
            enable_cors: true,
            cors_origins: vec!["*".to_string()],
        }
    }
}

/// Generic API response wrapper.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ApiResponse<T> {
    /// Whether the request was successful.
    pub success: bool,
    /// Response data (if successful).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<T>,
    /// Error message (if failed).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Response timestamp in milliseconds.
    pub timestamp: u64,
}

impl<T> ApiResponse<T> {
    /// Creates a successful response.
    pub fn success(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
            timestamp: current_timestamp(),
        }
    }

    /// Creates an error response.
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(message.into()),
            timestamp: current_timestamp(),
        }
    }
}

/// API error type.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ApiError {
    /// Error code.
    pub code: String,
    /// Error message.
    pub message: String,
}

impl ApiError {
    /// Creates a new API error.
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
        }
    }
}

/// Market maker status response.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct StatusResponse {
    /// Whether the market maker is running.
    pub running: bool,
    /// Whether quoting is enabled.
    pub quoting_enabled: bool,
    /// Current underlying symbol.
    pub symbol: String,
    /// Current underlying price.
    #[schema(value_type = f64)]
    pub underlying_price: Decimal,
    /// Uptime in seconds.
    pub uptime_seconds: u64,
    /// Number of active quotes.
    pub active_quotes: u32,
    /// Last quote update timestamp.
    pub last_quote_update: u64,
}

/// Configuration request for updating market maker settings.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ConfigRequest {
    /// Base spread in basis points.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_spread_bps: Option<u32>,
    /// Quote size.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(value_type = Option<f64>)]
    pub quote_size: Option<Decimal>,
    /// Risk aversion parameter (gamma).
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(value_type = Option<f64>)]
    pub risk_aversion: Option<Decimal>,
    /// Maximum delta limit.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(value_type = Option<f64>)]
    pub max_delta: Option<Decimal>,
    /// Maximum gamma limit.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(value_type = Option<f64>)]
    pub max_gamma: Option<Decimal>,
    /// Maximum vega limit.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(value_type = Option<f64>)]
    pub max_vega: Option<Decimal>,
    /// Enable/disable quoting.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quoting_enabled: Option<bool>,
}

/// Configuration response with current settings.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ConfigResponse {
    /// Base spread in basis points.
    pub base_spread_bps: u32,
    /// Quote size.
    #[schema(value_type = f64)]
    pub quote_size: Decimal,
    /// Risk aversion parameter (gamma).
    #[schema(value_type = f64)]
    pub risk_aversion: Decimal,
    /// Maximum delta limit.
    #[schema(value_type = f64)]
    pub max_delta: Decimal,
    /// Maximum gamma limit.
    #[schema(value_type = f64)]
    pub max_gamma: Decimal,
    /// Maximum vega limit.
    #[schema(value_type = f64)]
    pub max_vega: Decimal,
    /// Whether quoting is enabled.
    pub quoting_enabled: bool,
}

/// Quote response with current bid/ask.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct QuoteResponse {
    /// Trading symbol.
    pub symbol: String,
    /// Strike price (for options).
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(value_type = Option<f64>)]
    pub strike: Option<Decimal>,
    /// Bid price.
    #[schema(value_type = f64)]
    pub bid_price: Decimal,
    /// Bid size.
    #[schema(value_type = f64)]
    pub bid_size: Decimal,
    /// Ask price.
    #[schema(value_type = f64)]
    pub ask_price: Decimal,
    /// Ask size.
    #[schema(value_type = f64)]
    pub ask_size: Decimal,
    /// Theoretical value.
    #[schema(value_type = f64)]
    pub theo: Decimal,
    /// Spread in basis points.
    pub spread_bps: u32,
    /// Quote timestamp.
    pub timestamp: u64,
}

/// Position response with current inventory.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct PositionResponse {
    /// Trading symbol.
    pub symbol: String,
    /// Current position quantity.
    #[schema(value_type = f64)]
    pub quantity: Decimal,
    /// Average entry price.
    #[schema(value_type = f64)]
    pub avg_price: Decimal,
    /// Current market price.
    #[schema(value_type = f64)]
    pub market_price: Decimal,
    /// Unrealized P&L.
    #[schema(value_type = f64)]
    pub unrealized_pnl: Decimal,
    /// Realized P&L.
    #[schema(value_type = f64)]
    pub realized_pnl: Decimal,
    /// Position timestamp.
    pub timestamp: u64,
}

/// Greeks response with current risk metrics.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct GreeksResponse {
    /// Portfolio delta.
    #[schema(value_type = f64)]
    pub delta: Decimal,
    /// Portfolio gamma.
    #[schema(value_type = f64)]
    pub gamma: Decimal,
    /// Portfolio theta.
    #[schema(value_type = f64)]
    pub theta: Decimal,
    /// Portfolio vega.
    #[schema(value_type = f64)]
    pub vega: Decimal,
    /// Portfolio rho.
    #[schema(value_type = f64)]
    pub rho: Decimal,
    /// Delta utilization percentage.
    #[schema(value_type = f64)]
    pub delta_utilization: Decimal,
    /// Gamma utilization percentage.
    #[schema(value_type = f64)]
    pub gamma_utilization: Decimal,
    /// Vega utilization percentage.
    #[schema(value_type = f64)]
    pub vega_utilization: Decimal,
    /// Greeks timestamp.
    pub timestamp: u64,
}

/// P&L response with profit/loss breakdown.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct PnLResponse {
    /// Total P&L.
    #[schema(value_type = f64)]
    pub total_pnl: Decimal,
    /// Realized P&L.
    #[schema(value_type = f64)]
    pub realized_pnl: Decimal,
    /// Unrealized P&L.
    #[schema(value_type = f64)]
    pub unrealized_pnl: Decimal,
    /// Delta P&L attribution.
    #[schema(value_type = f64)]
    pub delta_pnl: Decimal,
    /// Gamma P&L attribution.
    #[schema(value_type = f64)]
    pub gamma_pnl: Decimal,
    /// Theta P&L attribution.
    #[schema(value_type = f64)]
    pub theta_pnl: Decimal,
    /// Vega P&L attribution.
    #[schema(value_type = f64)]
    pub vega_pnl: Decimal,
    /// Trading edge P&L.
    #[schema(value_type = f64)]
    pub edge_pnl: Decimal,
    /// P&L timestamp.
    pub timestamp: u64,
}

/// Returns current timestamp in milliseconds.
fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_api_response_success() {
        let response = ApiResponse::success("test data");
        assert!(response.success);
        assert_eq!(response.data, Some("test data"));
        assert!(response.error.is_none());
    }

    #[test]
    fn test_api_response_error() {
        let response: ApiResponse<String> = ApiResponse::error("test error");
        assert!(!response.success);
        assert!(response.data.is_none());
        assert_eq!(response.error, Some("test error".to_string()));
    }

    #[test]
    fn test_api_config_default() {
        let config = ApiConfig::default();
        assert_eq!(config.bind_address, "0.0.0.0:8080");
        assert!(config.enable_cors);
    }

    #[test]
    fn test_status_response() {
        let status = StatusResponse {
            running: true,
            quoting_enabled: true,
            symbol: "BTC".to_string(),
            underlying_price: dec!(50000.0),
            uptime_seconds: 3600,
            active_quotes: 10,
            last_quote_update: 1000,
        };
        assert!(status.running);
        assert_eq!(status.symbol, "BTC");
    }
}
