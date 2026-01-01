//! API state management.

use crate::Decimal;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use tokio::sync::RwLock;

use crate::api::types::{
    ConfigResponse, GreeksResponse, PnLResponse, PositionResponse, QuoteResponse, StatusResponse,
};

/// Shared API state for the market maker.
pub struct ApiState {
    /// Whether the market maker is running.
    running: AtomicBool,
    /// Whether quoting is enabled.
    quoting_enabled: AtomicBool,
    /// Start timestamp.
    start_time: AtomicU64,
    /// Current configuration.
    config: RwLock<ConfigState>,
    /// Current status.
    status: RwLock<StatusState>,
    /// Current quotes.
    quotes: RwLock<Vec<QuoteResponse>>,
    /// Current positions.
    positions: RwLock<Vec<PositionResponse>>,
    /// Current Greeks.
    greeks: RwLock<GreeksState>,
    /// Current P&L.
    pnl: RwLock<PnLState>,
}

/// Internal configuration state.
struct ConfigState {
    base_spread_bps: u32,
    quote_size: Decimal,
    risk_aversion: Decimal,
    max_delta: Decimal,
    max_gamma: Decimal,
    max_vega: Decimal,
}

/// Internal status state.
struct StatusState {
    symbol: String,
    underlying_price: Decimal,
    active_quotes: u32,
    last_quote_update: u64,
}

/// Internal Greeks state.
struct GreeksState {
    delta: Decimal,
    gamma: Decimal,
    theta: Decimal,
    vega: Decimal,
    rho: Decimal,
    delta_utilization: Decimal,
    gamma_utilization: Decimal,
    vega_utilization: Decimal,
}

/// Internal P&L state.
struct PnLState {
    total_pnl: Decimal,
    realized_pnl: Decimal,
    unrealized_pnl: Decimal,
    delta_pnl: Decimal,
    gamma_pnl: Decimal,
    theta_pnl: Decimal,
    vega_pnl: Decimal,
    edge_pnl: Decimal,
}

impl Default for ApiState {
    fn default() -> Self {
        Self::new()
    }
}

impl ApiState {
    /// Creates a new API state.
    #[must_use]
    pub fn new() -> Self {
        let now = current_timestamp();
        Self {
            running: AtomicBool::new(false),
            quoting_enabled: AtomicBool::new(false),
            start_time: AtomicU64::new(now),
            config: RwLock::new(ConfigState {
                base_spread_bps: 100,
                quote_size: Decimal::from(10),
                risk_aversion: Decimal::new(1, 1), // 0.1
                max_delta: Decimal::from(100),
                max_gamma: Decimal::from(50),
                max_vega: Decimal::from(1000),
            }),
            status: RwLock::new(StatusState {
                symbol: "BTC".to_string(),
                underlying_price: Decimal::ZERO,
                active_quotes: 0,
                last_quote_update: 0,
            }),
            quotes: RwLock::new(Vec::new()),
            positions: RwLock::new(Vec::new()),
            greeks: RwLock::new(GreeksState {
                delta: Decimal::ZERO,
                gamma: Decimal::ZERO,
                theta: Decimal::ZERO,
                vega: Decimal::ZERO,
                rho: Decimal::ZERO,
                delta_utilization: Decimal::ZERO,
                gamma_utilization: Decimal::ZERO,
                vega_utilization: Decimal::ZERO,
            }),
            pnl: RwLock::new(PnLState {
                total_pnl: Decimal::ZERO,
                realized_pnl: Decimal::ZERO,
                unrealized_pnl: Decimal::ZERO,
                delta_pnl: Decimal::ZERO,
                gamma_pnl: Decimal::ZERO,
                theta_pnl: Decimal::ZERO,
                vega_pnl: Decimal::ZERO,
                edge_pnl: Decimal::ZERO,
            }),
        }
    }

    /// Starts the market maker.
    pub fn start(&self) {
        self.running.store(true, Ordering::SeqCst);
        self.start_time.store(current_timestamp(), Ordering::SeqCst);
    }

    /// Stops the market maker.
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
        self.quoting_enabled.store(false, Ordering::SeqCst);
    }

    /// Enables quoting.
    pub fn enable_quoting(&self) {
        self.quoting_enabled.store(true, Ordering::SeqCst);
    }

    /// Disables quoting.
    pub fn disable_quoting(&self) {
        self.quoting_enabled.store(false, Ordering::SeqCst);
    }

    /// Returns whether the market maker is running.
    #[must_use]
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Returns whether quoting is enabled.
    #[must_use]
    pub fn is_quoting_enabled(&self) -> bool {
        self.quoting_enabled.load(Ordering::SeqCst)
    }

    /// Returns the uptime in seconds.
    #[must_use]
    pub fn uptime_seconds(&self) -> u64 {
        let start = self.start_time.load(Ordering::SeqCst);
        let now = current_timestamp();
        (now - start) / 1000
    }

    /// Gets the current status.
    pub async fn get_status(&self) -> StatusResponse {
        let status = self.status.read().await;
        StatusResponse {
            running: self.is_running(),
            quoting_enabled: self.is_quoting_enabled(),
            symbol: status.symbol.clone(),
            underlying_price: status.underlying_price,
            uptime_seconds: self.uptime_seconds(),
            active_quotes: status.active_quotes,
            last_quote_update: status.last_quote_update,
        }
    }

    /// Gets the current configuration.
    pub async fn get_config(&self) -> ConfigResponse {
        let config = self.config.read().await;
        ConfigResponse {
            base_spread_bps: config.base_spread_bps,
            quote_size: config.quote_size,
            risk_aversion: config.risk_aversion,
            max_delta: config.max_delta,
            max_gamma: config.max_gamma,
            max_vega: config.max_vega,
            quoting_enabled: self.is_quoting_enabled(),
        }
    }

    /// Updates the configuration.
    #[allow(clippy::too_many_arguments)]
    pub async fn update_config(
        &self,
        base_spread_bps: Option<u32>,
        quote_size: Option<Decimal>,
        risk_aversion: Option<Decimal>,
        max_delta: Option<Decimal>,
        max_gamma: Option<Decimal>,
        max_vega: Option<Decimal>,
        quoting_enabled: Option<bool>,
    ) {
        let mut config = self.config.write().await;
        if let Some(v) = base_spread_bps {
            config.base_spread_bps = v;
        }
        if let Some(v) = quote_size {
            config.quote_size = v;
        }
        if let Some(v) = risk_aversion {
            config.risk_aversion = v;
        }
        if let Some(v) = max_delta {
            config.max_delta = v;
        }
        if let Some(v) = max_gamma {
            config.max_gamma = v;
        }
        if let Some(v) = max_vega {
            config.max_vega = v;
        }
        drop(config);

        if let Some(enabled) = quoting_enabled {
            if enabled {
                self.enable_quoting();
            } else {
                self.disable_quoting();
            }
        }
    }

    /// Updates the underlying price.
    pub async fn update_underlying_price(&self, price: Decimal) {
        let mut status = self.status.write().await;
        status.underlying_price = price;
    }

    /// Updates the symbol.
    pub async fn update_symbol(&self, symbol: String) {
        let mut status = self.status.write().await;
        status.symbol = symbol;
    }

    /// Gets all current quotes.
    pub async fn get_quotes(&self) -> Vec<QuoteResponse> {
        self.quotes.read().await.clone()
    }

    /// Updates the quotes.
    pub async fn update_quotes(&self, quotes: Vec<QuoteResponse>) {
        let mut status = self.status.write().await;
        status.active_quotes = quotes.len() as u32;
        status.last_quote_update = current_timestamp();
        drop(status);

        let mut q = self.quotes.write().await;
        *q = quotes;
    }

    /// Gets all current positions.
    pub async fn get_positions(&self) -> Vec<PositionResponse> {
        self.positions.read().await.clone()
    }

    /// Updates the positions.
    pub async fn update_positions(&self, positions: Vec<PositionResponse>) {
        let mut p = self.positions.write().await;
        *p = positions;
    }

    /// Gets the current Greeks.
    pub async fn get_greeks(&self) -> GreeksResponse {
        let greeks = self.greeks.read().await;
        GreeksResponse {
            delta: greeks.delta,
            gamma: greeks.gamma,
            theta: greeks.theta,
            vega: greeks.vega,
            rho: greeks.rho,
            delta_utilization: greeks.delta_utilization,
            gamma_utilization: greeks.gamma_utilization,
            vega_utilization: greeks.vega_utilization,
            timestamp: current_timestamp(),
        }
    }

    /// Updates the Greeks.
    #[allow(clippy::too_many_arguments)]
    pub async fn update_greeks(
        &self,
        delta: Decimal,
        gamma: Decimal,
        theta: Decimal,
        vega: Decimal,
        rho: Decimal,
        delta_utilization: Decimal,
        gamma_utilization: Decimal,
        vega_utilization: Decimal,
    ) {
        let mut g = self.greeks.write().await;
        g.delta = delta;
        g.gamma = gamma;
        g.theta = theta;
        g.vega = vega;
        g.rho = rho;
        g.delta_utilization = delta_utilization;
        g.gamma_utilization = gamma_utilization;
        g.vega_utilization = vega_utilization;
    }

    /// Gets the current P&L.
    pub async fn get_pnl(&self) -> PnLResponse {
        let pnl = self.pnl.read().await;
        PnLResponse {
            total_pnl: pnl.total_pnl,
            realized_pnl: pnl.realized_pnl,
            unrealized_pnl: pnl.unrealized_pnl,
            delta_pnl: pnl.delta_pnl,
            gamma_pnl: pnl.gamma_pnl,
            theta_pnl: pnl.theta_pnl,
            vega_pnl: pnl.vega_pnl,
            edge_pnl: pnl.edge_pnl,
            timestamp: current_timestamp(),
        }
    }

    /// Updates the P&L.
    #[allow(clippy::too_many_arguments)]
    pub async fn update_pnl(
        &self,
        total_pnl: Decimal,
        realized_pnl: Decimal,
        unrealized_pnl: Decimal,
        delta_pnl: Decimal,
        gamma_pnl: Decimal,
        theta_pnl: Decimal,
        vega_pnl: Decimal,
        edge_pnl: Decimal,
    ) {
        let mut p = self.pnl.write().await;
        p.total_pnl = total_pnl;
        p.realized_pnl = realized_pnl;
        p.unrealized_pnl = unrealized_pnl;
        p.delta_pnl = delta_pnl;
        p.gamma_pnl = gamma_pnl;
        p.theta_pnl = theta_pnl;
        p.vega_pnl = vega_pnl;
        p.edge_pnl = edge_pnl;
    }
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

    #[tokio::test]
    async fn test_api_state_new() {
        let state = ApiState::new();
        assert!(!state.is_running());
        assert!(!state.is_quoting_enabled());
    }

    #[tokio::test]
    async fn test_api_state_start_stop() {
        let state = ApiState::new();

        state.start();
        assert!(state.is_running());

        state.stop();
        assert!(!state.is_running());
    }

    #[tokio::test]
    async fn test_api_state_quoting() {
        let state = ApiState::new();

        state.enable_quoting();
        assert!(state.is_quoting_enabled());

        state.disable_quoting();
        assert!(!state.is_quoting_enabled());
    }

    #[tokio::test]
    async fn test_api_state_config() {
        let state = ApiState::new();

        state
            .update_config(
                Some(200),
                Some(dec!(20.0)),
                None,
                None,
                None,
                None,
                Some(true),
            )
            .await;

        let config = state.get_config().await;
        assert_eq!(config.base_spread_bps, 200);
        assert_eq!(config.quote_size, dec!(20.0));
        assert!(config.quoting_enabled);
    }

    #[tokio::test]
    async fn test_api_state_status() {
        let state = ApiState::new();
        state.start();
        state.update_symbol("ETH".to_string()).await;
        state.update_underlying_price(dec!(3000.0)).await;

        let status = state.get_status().await;
        assert!(status.running);
        assert_eq!(status.symbol, "ETH");
        assert_eq!(status.underlying_price, dec!(3000.0));
    }

    #[tokio::test]
    async fn test_api_state_greeks() {
        let state = ApiState::new();

        state
            .update_greeks(
                dec!(50.0),
                dec!(10.0),
                dec!(-5.0),
                dec!(100.0),
                dec!(2.0),
                dec!(0.5),
                dec!(0.2),
                dec!(0.1),
            )
            .await;

        let greeks = state.get_greeks().await;
        assert_eq!(greeks.delta, dec!(50.0));
        assert_eq!(greeks.gamma, dec!(10.0));
    }
}
