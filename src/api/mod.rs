//! REST/WebSocket API module for market maker control and monitoring.
//!
//! This module provides HTTP endpoints for controlling and monitoring the market maker,
//! including configuration, status, quotes, positions, and Greeks.
//!
//! # Feature Flag
//!
//! Enable with:
//! ```toml
//! [dependencies]
//! market-maker-rs = { version = "0.3", features = ["api"] }
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use market_maker_rs::api::{ApiConfig, create_router, ApiState};
//! use std::sync::Arc;
//!
//! let config = ApiConfig::default();
//! let state = Arc::new(ApiState::new());
//! let router = create_router(state);
//!
//! // Run with axum
//! let listener = tokio::net::TcpListener::bind(&config.bind_address).await?;
//! axum::serve(listener, router).await?;
//! ```

mod handlers;
mod routes;
mod state;
mod types;

pub use routes::{ApiDoc, create_router, openapi_json};
pub use state::ApiState;
pub use types::{
    ApiConfig, ApiError, ApiResponse, ConfigRequest, ConfigResponse, GreeksResponse, PnLResponse,
    PositionResponse, QuoteResponse, StatusResponse,
};
