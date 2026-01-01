//! API request handlers.

use axum::{Json, extract::State, http::StatusCode};
use std::sync::Arc;

use crate::api::state::ApiState;
use crate::api::types::{
    ApiResponse, ConfigRequest, ConfigResponse, GreeksResponse, PnLResponse, PositionResponse,
    QuoteResponse, StatusResponse,
};

/// Health check handler.
#[utoipa::path(
    get,
    path = "/health",
    responses(
        (status = 200, description = "Service is healthy", body = ApiResponse<String>)
    ),
    tag = "health"
)]
pub async fn health() -> Json<ApiResponse<String>> {
    Json(ApiResponse::success("ok".to_string()))
}

/// Get market maker status.
#[utoipa::path(
    get,
    path = "/api/v1/status",
    responses(
        (status = 200, description = "Current status", body = ApiResponse<StatusResponse>)
    ),
    tag = "market-maker"
)]
pub async fn get_status(State(state): State<Arc<ApiState>>) -> Json<ApiResponse<StatusResponse>> {
    let status = state.get_status().await;
    Json(ApiResponse::success(status))
}

/// Start the market maker.
#[utoipa::path(
    post,
    path = "/api/v1/start",
    responses(
        (status = 200, description = "Market maker started", body = ApiResponse<String>)
    ),
    tag = "market-maker"
)]
pub async fn start(State(state): State<Arc<ApiState>>) -> Json<ApiResponse<String>> {
    state.start();
    Json(ApiResponse::success("Market maker started".to_string()))
}

/// Stop the market maker.
#[utoipa::path(
    post,
    path = "/api/v1/stop",
    responses(
        (status = 200, description = "Market maker stopped", body = ApiResponse<String>)
    ),
    tag = "market-maker"
)]
pub async fn stop(State(state): State<Arc<ApiState>>) -> Json<ApiResponse<String>> {
    state.stop();
    Json(ApiResponse::success("Market maker stopped".to_string()))
}

/// Get current configuration.
#[utoipa::path(
    get,
    path = "/api/v1/config",
    responses(
        (status = 200, description = "Current configuration", body = ApiResponse<ConfigResponse>)
    ),
    tag = "market-maker"
)]
pub async fn get_config(State(state): State<Arc<ApiState>>) -> Json<ApiResponse<ConfigResponse>> {
    let config = state.get_config().await;
    Json(ApiResponse::success(config))
}

/// Update configuration.
#[utoipa::path(
    put,
    path = "/api/v1/config",
    request_body = ConfigRequest,
    responses(
        (status = 200, description = "Configuration updated", body = ApiResponse<ConfigResponse>)
    ),
    tag = "market-maker"
)]
pub async fn update_config(
    State(state): State<Arc<ApiState>>,
    Json(request): Json<ConfigRequest>,
) -> Json<ApiResponse<ConfigResponse>> {
    state
        .update_config(
            request.base_spread_bps,
            request.quote_size,
            request.risk_aversion,
            request.max_delta,
            request.max_gamma,
            request.max_vega,
            request.quoting_enabled,
        )
        .await;

    let config = state.get_config().await;
    Json(ApiResponse::success(config))
}

/// Get current quotes.
#[utoipa::path(
    get,
    path = "/api/v1/quotes",
    responses(
        (status = 200, description = "Current quotes", body = ApiResponse<Vec<QuoteResponse>>)
    ),
    tag = "market-maker"
)]
pub async fn get_quotes(
    State(state): State<Arc<ApiState>>,
) -> Json<ApiResponse<Vec<QuoteResponse>>> {
    let quotes = state.get_quotes().await;
    Json(ApiResponse::success(quotes))
}

/// Get current positions.
#[utoipa::path(
    get,
    path = "/api/v1/positions",
    responses(
        (status = 200, description = "Current positions", body = ApiResponse<Vec<PositionResponse>>)
    ),
    tag = "market-maker"
)]
pub async fn get_positions(
    State(state): State<Arc<ApiState>>,
) -> Json<ApiResponse<Vec<PositionResponse>>> {
    let positions = state.get_positions().await;
    Json(ApiResponse::success(positions))
}

/// Get current Greeks.
#[utoipa::path(
    get,
    path = "/api/v1/greeks",
    responses(
        (status = 200, description = "Current Greeks", body = ApiResponse<GreeksResponse>)
    ),
    tag = "market-maker"
)]
pub async fn get_greeks(State(state): State<Arc<ApiState>>) -> Json<ApiResponse<GreeksResponse>> {
    let greeks = state.get_greeks().await;
    Json(ApiResponse::success(greeks))
}

/// Get current P&L.
#[utoipa::path(
    get,
    path = "/api/v1/pnl",
    responses(
        (status = 200, description = "Current P&L", body = ApiResponse<PnLResponse>)
    ),
    tag = "market-maker"
)]
pub async fn get_pnl(State(state): State<Arc<ApiState>>) -> Json<ApiResponse<PnLResponse>> {
    let pnl = state.get_pnl().await;
    Json(ApiResponse::success(pnl))
}

/// Enable quoting.
#[utoipa::path(
    post,
    path = "/api/v1/quoting/enable",
    responses(
        (status = 200, description = "Quoting enabled", body = ApiResponse<String>)
    ),
    tag = "market-maker"
)]
pub async fn enable_quoting(State(state): State<Arc<ApiState>>) -> Json<ApiResponse<String>> {
    state.enable_quoting();
    Json(ApiResponse::success("Quoting enabled".to_string()))
}

/// Disable quoting.
#[utoipa::path(
    post,
    path = "/api/v1/quoting/disable",
    responses(
        (status = 200, description = "Quoting disabled", body = ApiResponse<String>)
    ),
    tag = "market-maker"
)]
pub async fn disable_quoting(State(state): State<Arc<ApiState>>) -> Json<ApiResponse<String>> {
    state.disable_quoting();
    Json(ApiResponse::success("Quoting disabled".to_string()))
}

/// Fallback handler for 404.
pub async fn not_found() -> (StatusCode, Json<ApiResponse<()>>) {
    (StatusCode::NOT_FOUND, Json(ApiResponse::error("Not found")))
}
