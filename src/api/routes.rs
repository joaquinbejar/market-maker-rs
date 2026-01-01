//! API route configuration.

use axum::{
    Router,
    routing::{get, post, put},
};
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};
use utoipa::OpenApi;

use crate::api::handlers;
use crate::api::state::ApiState;
use crate::api::types::{
    ApiResponse, ConfigRequest, ConfigResponse, GreeksResponse, PnLResponse, PositionResponse,
    QuoteResponse, StatusResponse,
};

/// OpenAPI documentation.
#[derive(OpenApi)]
#[openapi(
    paths(
        handlers::health,
        handlers::get_status,
        handlers::start,
        handlers::stop,
        handlers::get_config,
        handlers::update_config,
        handlers::get_quotes,
        handlers::get_positions,
        handlers::get_greeks,
        handlers::get_pnl,
        handlers::enable_quoting,
        handlers::disable_quoting,
    ),
    components(schemas(
        ApiResponse<String>,
        ApiResponse<StatusResponse>,
        ApiResponse<ConfigResponse>,
        ApiResponse<Vec<QuoteResponse>>,
        ApiResponse<Vec<PositionResponse>>,
        ApiResponse<GreeksResponse>,
        ApiResponse<PnLResponse>,
        StatusResponse,
        ConfigRequest,
        ConfigResponse,
        QuoteResponse,
        PositionResponse,
        GreeksResponse,
        PnLResponse,
    )),
    tags(
        (name = "health", description = "Health check endpoints"),
        (name = "market-maker", description = "Market maker control and monitoring")
    ),
    info(
        title = "Market Maker API",
        version = "1.0.0",
        description = "REST API for controlling and monitoring the market maker"
    )
)]
pub struct ApiDoc;

/// Creates the API router with all routes.
pub fn create_router(state: Arc<ApiState>) -> Router {
    // CORS configuration
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    // Build router
    Router::new()
        // Health check
        .route("/health", get(handlers::health))
        // Status endpoints
        .route("/api/v1/status", get(handlers::get_status))
        .route("/api/v1/start", post(handlers::start))
        .route("/api/v1/stop", post(handlers::stop))
        // Configuration endpoints
        .route("/api/v1/config", get(handlers::get_config))
        .route("/api/v1/config", put(handlers::update_config))
        // Quote endpoints
        .route("/api/v1/quotes", get(handlers::get_quotes))
        // Position endpoints
        .route("/api/v1/positions", get(handlers::get_positions))
        // Greeks endpoints
        .route("/api/v1/greeks", get(handlers::get_greeks))
        // P&L endpoints
        .route("/api/v1/pnl", get(handlers::get_pnl))
        // Quoting control
        .route("/api/v1/quoting/enable", post(handlers::enable_quoting))
        .route("/api/v1/quoting/disable", post(handlers::disable_quoting))
        // Fallback
        .fallback(handlers::not_found)
        // State
        .with_state(state)
        // CORS
        .layer(cors)
}

/// Returns the OpenAPI JSON documentation.
#[must_use]
pub fn openapi_json() -> String {
    ApiDoc::openapi().to_pretty_json().unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_router() {
        let state = Arc::new(ApiState::new());
        let _router = create_router(state);
    }

    #[test]
    fn test_openapi_doc() {
        let doc = ApiDoc::openapi();
        assert_eq!(doc.info.title, "Market Maker API");
        assert_eq!(doc.info.version, "1.0.0");
    }
}
