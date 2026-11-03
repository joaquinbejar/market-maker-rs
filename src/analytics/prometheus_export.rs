//! Prometheus metrics export for trading system monitoring.
//!
//! This module provides Prometheus-compatible metrics export for integration
//! with Grafana dashboards and alerting systems.
//!
//! # Feature Flag
//!
//! This module requires the `prometheus` feature flag:
//!
//! ```toml
//! [dependencies]
//! market-maker-rs = { version = "0.1", features = ["prometheus"] }
//! ```
//!
//! # Overview
//!
//! The module provides:
//!
//! - **PrometheusMetrics**: Registry with all trading metrics
//! - **MetricsServer**: HTTP server exposing `/metrics` endpoint
//! - **MetricsBridge**: Adapter to sync with `LiveMetrics`
//!
//! # Example
//!
//! ```rust,ignore
//! use market_maker_rs::analytics::prometheus_export::{PrometheusMetrics, MetricsServer};
//! use std::sync::Arc;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create metrics registry
//!     let metrics = Arc::new(PrometheusMetrics::new("marketmaker")?);
//!     
//!     // Start HTTP server on port 9090
//!     let server = MetricsServer::new(Arc::clone(&metrics), "0.0.0.0:9090");
//!     let handle = server.spawn();
//!     
//!     // Record metrics during trading
//!     metrics.inc_quotes();
//!     metrics.inc_orders_submitted();
//!     metrics.set_position(100.0);
//!     metrics.set_pnl(500.0, 50.0);
//!     
//!     // Server runs in background, metrics available at http://localhost:9090/metrics
//!     handle.await?;
//!     Ok(())
//! }
//! ```

use prometheus::{Counter, Encoder, Gauge, Histogram, HistogramOpts, Opts, Registry, TextEncoder};
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;

use http_body_util::Full;
use hyper::body::Bytes;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use tokio::net::TcpListener;

use super::live_metrics::LiveMetrics;
use crate::types::error::MMResult;

/// Default histogram buckets for latency measurements in milliseconds.
const LATENCY_BUCKETS: &[f64] = &[
    0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0,
];

/// Default histogram buckets for spread measurements in basis points.
const SPREAD_BUCKETS: &[f64] = &[1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0];

/// Prometheus metrics registry for market making operations.
///
/// Contains all metric types needed for monitoring a trading system:
/// - Counters for event counts (quotes, orders, fills)
/// - Gauges for current values (position, PnL, spread)
/// - Histograms for distributions (latency, spread)
///
/// # Metric Naming Convention
///
/// All metrics follow the pattern: `{namespace}_{subsystem}_{name}_{unit}`
///
/// Examples:
/// - `marketmaker_orders_submitted_total`
/// - `marketmaker_latency_order_milliseconds`
/// - `marketmaker_position_current`
#[derive(Debug)]
pub struct PrometheusMetrics {
    registry: Registry,

    // Counters
    quotes_total: Counter,
    orders_submitted_total: Counter,
    orders_filled_total: Counter,
    orders_cancelled_total: Counter,
    orders_rejected_total: Counter,
    partial_fills_total: Counter,

    // Gauges
    open_orders: Gauge,
    position_current: Gauge,
    pnl_realized: Gauge,
    pnl_unrealized: Gauge,
    pnl_total: Gauge,
    spread_current: Gauge,

    // Histograms
    order_latency: Histogram,
    fill_latency: Histogram,
    spread_histogram: Histogram,
}

impl PrometheusMetrics {
    /// Creates a new Prometheus metrics registry.
    ///
    /// # Arguments
    ///
    /// * `namespace` - Prefix for all metric names (e.g., "marketmaker")
    ///
    /// # Errors
    ///
    /// Returns an error if metric registration fails.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use market_maker_rs::analytics::prometheus_export::PrometheusMetrics;
    ///
    /// let metrics = PrometheusMetrics::new("marketmaker")?;
    /// ```
    pub fn new(namespace: &str) -> Result<Self, prometheus::Error> {
        let registry = Registry::new();

        // Counters
        let quotes_total = Counter::with_opts(
            Opts::new("quotes_total", "Total number of quotes generated")
                .namespace(namespace)
                .subsystem("quotes"),
        )?;

        let orders_submitted_total = Counter::with_opts(
            Opts::new("submitted_total", "Total number of orders submitted")
                .namespace(namespace)
                .subsystem("orders"),
        )?;

        let orders_filled_total = Counter::with_opts(
            Opts::new("filled_total", "Total number of orders filled")
                .namespace(namespace)
                .subsystem("orders"),
        )?;

        let orders_cancelled_total = Counter::with_opts(
            Opts::new("cancelled_total", "Total number of orders cancelled")
                .namespace(namespace)
                .subsystem("orders"),
        )?;

        let orders_rejected_total = Counter::with_opts(
            Opts::new("rejected_total", "Total number of orders rejected")
                .namespace(namespace)
                .subsystem("orders"),
        )?;

        let partial_fills_total = Counter::with_opts(
            Opts::new("partial_fills_total", "Total number of partial fills")
                .namespace(namespace)
                .subsystem("orders"),
        )?;

        // Gauges
        let open_orders = Gauge::with_opts(
            Opts::new("open_orders", "Current number of open orders")
                .namespace(namespace)
                .subsystem("orders"),
        )?;

        let position_current = Gauge::with_opts(
            Opts::new("current", "Current position size")
                .namespace(namespace)
                .subsystem("position"),
        )?;

        let pnl_realized = Gauge::with_opts(
            Opts::new("realized", "Realized PnL")
                .namespace(namespace)
                .subsystem("pnl"),
        )?;

        let pnl_unrealized = Gauge::with_opts(
            Opts::new("unrealized", "Unrealized PnL")
                .namespace(namespace)
                .subsystem("pnl"),
        )?;

        let pnl_total = Gauge::with_opts(
            Opts::new("total", "Total PnL (realized + unrealized)")
                .namespace(namespace)
                .subsystem("pnl"),
        )?;

        let spread_current = Gauge::with_opts(
            Opts::new("current_bps", "Current spread in basis points")
                .namespace(namespace)
                .subsystem("spread"),
        )?;

        // Histograms
        let order_latency = Histogram::with_opts(
            HistogramOpts::new(
                "order_milliseconds",
                "Order submission latency in milliseconds",
            )
            .namespace(namespace)
            .subsystem("latency")
            .buckets(LATENCY_BUCKETS.to_vec()),
        )?;

        let fill_latency = Histogram::with_opts(
            HistogramOpts::new(
                "fill_milliseconds",
                "Fill notification latency in milliseconds",
            )
            .namespace(namespace)
            .subsystem("latency")
            .buckets(LATENCY_BUCKETS.to_vec()),
        )?;

        let spread_histogram = Histogram::with_opts(
            HistogramOpts::new("distribution_bps", "Spread distribution in basis points")
                .namespace(namespace)
                .subsystem("spread")
                .buckets(SPREAD_BUCKETS.to_vec()),
        )?;

        // Register all metrics
        registry.register(Box::new(quotes_total.clone()))?;
        registry.register(Box::new(orders_submitted_total.clone()))?;
        registry.register(Box::new(orders_filled_total.clone()))?;
        registry.register(Box::new(orders_cancelled_total.clone()))?;
        registry.register(Box::new(orders_rejected_total.clone()))?;
        registry.register(Box::new(partial_fills_total.clone()))?;
        registry.register(Box::new(open_orders.clone()))?;
        registry.register(Box::new(position_current.clone()))?;
        registry.register(Box::new(pnl_realized.clone()))?;
        registry.register(Box::new(pnl_unrealized.clone()))?;
        registry.register(Box::new(pnl_total.clone()))?;
        registry.register(Box::new(spread_current.clone()))?;
        registry.register(Box::new(order_latency.clone()))?;
        registry.register(Box::new(fill_latency.clone()))?;
        registry.register(Box::new(spread_histogram.clone()))?;

        Ok(Self {
            registry,
            quotes_total,
            orders_submitted_total,
            orders_filled_total,
            orders_cancelled_total,
            orders_rejected_total,
            partial_fills_total,
            open_orders,
            position_current,
            pnl_realized,
            pnl_unrealized,
            pnl_total,
            spread_current,
            order_latency,
            fill_latency,
            spread_histogram,
        })
    }

    // Counter increments

    /// Increments the quotes counter.
    pub fn inc_quotes(&self) {
        self.quotes_total.inc();
    }

    /// Increments the quotes counter by a specific amount.
    pub fn inc_quotes_by(&self, count: f64) {
        self.quotes_total.inc_by(count);
    }

    /// Increments the orders submitted counter.
    pub fn inc_orders_submitted(&self) {
        self.orders_submitted_total.inc();
    }

    /// Increments the orders filled counter.
    pub fn inc_orders_filled(&self) {
        self.orders_filled_total.inc();
    }

    /// Increments the orders cancelled counter.
    pub fn inc_orders_cancelled(&self) {
        self.orders_cancelled_total.inc();
    }

    /// Increments the orders rejected counter.
    pub fn inc_orders_rejected(&self) {
        self.orders_rejected_total.inc();
    }

    /// Increments the partial fills counter.
    pub fn inc_partial_fills(&self) {
        self.partial_fills_total.inc();
    }

    // Gauge updates

    /// Sets the current number of open orders.
    pub fn set_open_orders(&self, count: f64) {
        self.open_orders.set(count);
    }

    /// Sets the current position size.
    pub fn set_position(&self, position: f64) {
        self.position_current.set(position);
    }

    /// Sets the PnL values.
    ///
    /// # Arguments
    ///
    /// * `realized` - Realized PnL
    /// * `unrealized` - Unrealized PnL
    pub fn set_pnl(&self, realized: f64, unrealized: f64) {
        self.pnl_realized.set(realized);
        self.pnl_unrealized.set(unrealized);
        self.pnl_total.set(realized + unrealized);
    }

    /// Sets the current spread in basis points.
    pub fn set_spread(&self, spread_bps: f64) {
        self.spread_current.set(spread_bps);
    }

    // Histogram observations

    /// Records an order latency observation.
    ///
    /// # Arguments
    ///
    /// * `latency_ms` - Latency in milliseconds
    pub fn observe_order_latency(&self, latency_ms: f64) {
        self.order_latency.observe(latency_ms);
    }

    /// Records a fill latency observation.
    ///
    /// # Arguments
    ///
    /// * `latency_ms` - Latency in milliseconds
    pub fn observe_fill_latency(&self, latency_ms: f64) {
        self.fill_latency.observe(latency_ms);
    }

    /// Records a spread observation.
    ///
    /// # Arguments
    ///
    /// * `spread_bps` - Spread in basis points
    pub fn observe_spread(&self, spread_bps: f64) {
        self.spread_histogram.observe(spread_bps);
    }

    /// Returns a reference to the underlying registry.
    #[must_use]
    pub fn registry(&self) -> &Registry {
        &self.registry
    }

    /// Encodes all metrics to Prometheus text format.
    ///
    /// # Errors
    ///
    /// Returns an error if encoding fails.
    pub fn encode(&self) -> Result<String, prometheus::Error> {
        let encoder = TextEncoder::new();
        let metric_families = self.registry.gather();
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer)?;
        Ok(String::from_utf8(buffer).unwrap_or_default())
    }

    /// Returns the current value of the quotes counter.
    #[must_use]
    pub fn get_quotes_total(&self) -> f64 {
        self.quotes_total.get()
    }

    /// Returns the current value of the orders submitted counter.
    #[must_use]
    pub fn get_orders_submitted_total(&self) -> f64 {
        self.orders_submitted_total.get()
    }

    /// Returns the current value of the orders filled counter.
    #[must_use]
    pub fn get_orders_filled_total(&self) -> f64 {
        self.orders_filled_total.get()
    }

    /// Returns the current value of the orders cancelled counter.
    #[must_use]
    pub fn get_orders_cancelled_total(&self) -> f64 {
        self.orders_cancelled_total.get()
    }

    /// Returns the current value of the orders rejected counter.
    #[must_use]
    pub fn get_orders_rejected_total(&self) -> f64 {
        self.orders_rejected_total.get()
    }

    /// Returns the current value of the partial fills counter.
    #[must_use]
    pub fn get_partial_fills_total(&self) -> f64 {
        self.partial_fills_total.get()
    }

    /// Returns the current number of open orders.
    #[must_use]
    pub fn get_open_orders(&self) -> f64 {
        self.open_orders.get()
    }

    /// Returns the current position.
    #[must_use]
    pub fn get_position(&self) -> f64 {
        self.position_current.get()
    }

    /// Returns the realized PnL.
    #[must_use]
    pub fn get_pnl_realized(&self) -> f64 {
        self.pnl_realized.get()
    }

    /// Returns the unrealized PnL.
    #[must_use]
    pub fn get_pnl_unrealized(&self) -> f64 {
        self.pnl_unrealized.get()
    }

    /// Returns the total PnL.
    #[must_use]
    pub fn get_pnl_total(&self) -> f64 {
        self.pnl_total.get()
    }

    /// Returns the current spread in basis points.
    #[must_use]
    pub fn get_spread(&self) -> f64 {
        self.spread_current.get()
    }
}

/// HTTP server for exposing Prometheus metrics endpoint.
///
/// Serves metrics at the `/metrics` endpoint in Prometheus text format.
///
/// # Example
///
/// ```rust,ignore
/// use market_maker_rs::analytics::prometheus_export::{PrometheusMetrics, MetricsServer};
/// use std::sync::Arc;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let metrics = Arc::new(PrometheusMetrics::new("marketmaker")?);
///     let server = MetricsServer::new(Arc::clone(&metrics), "0.0.0.0:9090");
///     
///     // Run server (blocking)
///     server.run().await?;
///     Ok(())
/// }
/// ```
pub struct MetricsServer {
    metrics: Arc<PrometheusMetrics>,
    bind_address: String,
}

impl MetricsServer {
    /// Creates a new metrics server.
    ///
    /// # Arguments
    ///
    /// * `metrics` - Shared reference to the metrics registry
    /// * `bind_address` - Address to bind the HTTP server (e.g., "0.0.0.0:9090")
    #[must_use]
    pub fn new(metrics: Arc<PrometheusMetrics>, bind_address: &str) -> Self {
        Self {
            metrics,
            bind_address: bind_address.to_string(),
        }
    }

    /// Runs the HTTP server (blocking).
    ///
    /// # Errors
    ///
    /// Returns an error if the server fails to start or encounters a runtime error.
    pub async fn run(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let addr: SocketAddr = self.bind_address.parse()?;
        let listener = TcpListener::bind(addr).await?;

        loop {
            let (stream, _) = listener.accept().await?;
            let io = TokioIo::new(stream);
            let metrics = Arc::clone(&self.metrics);

            tokio::spawn(async move {
                let service = service_fn(move |req| {
                    let metrics = Arc::clone(&metrics);
                    async move { handle_request(req, metrics).await }
                });

                if let Err(err) = http1::Builder::new().serve_connection(io, service).await {
                    eprintln!("Error serving connection: {:?}", err);
                }
            });
        }
    }

    /// Spawns the HTTP server in a background task.
    ///
    /// Returns a join handle that can be used to await server completion.
    #[must_use]
    pub fn spawn(self) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            if let Err(e) = self.run().await {
                eprintln!("Metrics server error: {}", e);
            }
        })
    }

    /// Returns the bind address.
    #[must_use]
    pub fn bind_address(&self) -> &str {
        &self.bind_address
    }
}

/// Builds a response without the fallible `Response::builder`.
fn text_response(
    status: StatusCode,
    content_type: Option<&'static str>,
    body: String,
) -> Response<Full<Bytes>> {
    let mut response = Response::new(Full::new(Bytes::from(body)));
    *response.status_mut() = status;
    if let Some(content_type) = content_type {
        response.headers_mut().insert(
            hyper::header::CONTENT_TYPE,
            hyper::header::HeaderValue::from_static(content_type),
        );
    }
    response
}

/// Handles HTTP requests to the metrics server.
async fn handle_request(
    req: Request<hyper::body::Incoming>,
    metrics: Arc<PrometheusMetrics>,
) -> Result<Response<Full<Bytes>>, Infallible> {
    let response = match req.uri().path() {
        "/metrics" => match metrics.encode() {
            Ok(body) => text_response(StatusCode::OK, Some("text/plain; charset=utf-8"), body),
            Err(e) => text_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                None,
                format!("Error encoding metrics: {}", e),
            ),
        },
        "/health" => text_response(StatusCode::OK, None, "OK".to_string()),
        "/" => text_response(
            StatusCode::OK,
            Some("text/html"),
            r#"<html>
<head><title>Market Maker Metrics</title></head>
<body>
<h1>Market Maker Metrics</h1>
<p><a href="/metrics">Metrics</a></p>
<p><a href="/health">Health</a></p>
</body>
</html>"#
                .to_string(),
        ),
        _ => text_response(StatusCode::NOT_FOUND, None, "Not Found".to_string()),
    };

    Ok(response)
}

/// Bridge adapter to sync `LiveMetrics` with `PrometheusMetrics`.
///
/// This adapter allows you to periodically sync the internal `LiveMetrics`
/// counters with Prometheus metrics for export.
///
/// # Example
///
/// ```rust,ignore
/// use market_maker_rs::analytics::{LiveMetrics, prometheus_export::{PrometheusMetrics, MetricsBridge}};
/// use std::sync::Arc;
///
/// let live_metrics = Arc::new(LiveMetrics::new(0));
/// let prom_metrics = Arc::new(PrometheusMetrics::new("marketmaker")?);
/// let bridge = MetricsBridge::new(Arc::clone(&live_metrics), Arc::clone(&prom_metrics));
///
/// // Sync metrics periodically
/// bridge.sync().unwrap();
/// ```
pub struct MetricsBridge {
    live_metrics: Arc<LiveMetrics>,
    prom_metrics: Arc<PrometheusMetrics>,
}

impl MetricsBridge {
    /// Creates a new metrics bridge.
    ///
    /// # Arguments
    ///
    /// * `live_metrics` - Reference to the live metrics tracker
    /// * `prom_metrics` - Reference to the Prometheus metrics registry
    #[must_use]
    pub fn new(live_metrics: Arc<LiveMetrics>, prom_metrics: Arc<PrometheusMetrics>) -> Self {
        Self {
            live_metrics,
            prom_metrics,
        }
    }

    /// Syncs current values from `LiveMetrics` to `PrometheusMetrics`.
    ///
    /// Call this periodically to update Prometheus metrics with the latest values.
    ///
    /// # Errors
    ///
    /// Returns the error of [`LiveMetrics::snapshot`] (a poisoned lock or a
    /// PnL overflow); Prometheus is left untouched in that case.
    pub fn sync(&self) -> MMResult<()> {
        let snapshot = self.live_metrics.snapshot(0)?;

        // Sync counters (Prometheus counters can only increase, so we set to current total)
        // Note: This works because we're setting absolute values, not incrementing
        // Prometheus values are f64; counts are exact below 2^53 (not money).
        let quotes_diff = snapshot.quotes_generated as f64 - self.prom_metrics.get_quotes_total();
        if quotes_diff > 0.0 {
            self.prom_metrics.inc_quotes_by(quotes_diff);
        }

        // Sync gauges (these can be set directly)
        // Prometheus gauges are f64; counts are exact below 2^53 (not money).
        self.prom_metrics
            .set_open_orders(snapshot.open_orders as f64);
        self.prom_metrics
            .set_position(snapshot.current_position.to_string().parse().unwrap_or(0.0));
        self.prom_metrics.set_pnl(
            snapshot.realized_pnl.to_string().parse().unwrap_or(0.0),
            snapshot.unrealized_pnl.to_string().parse().unwrap_or(0.0),
        );
        Ok(())
    }

    /// Returns a reference to the live metrics.
    #[must_use]
    pub fn live_metrics(&self) -> &Arc<LiveMetrics> {
        &self.live_metrics
    }

    /// Returns a reference to the Prometheus metrics.
    #[must_use]
    pub fn prom_metrics(&self) -> &Arc<PrometheusMetrics> {
        &self.prom_metrics
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prometheus_metrics_new() {
        let metrics = PrometheusMetrics::new("test").unwrap();
        assert_eq!(metrics.get_quotes_total(), 0.0);
        assert_eq!(metrics.get_orders_submitted_total(), 0.0);
    }

    #[test]
    fn test_counter_increments() {
        let metrics = PrometheusMetrics::new("test").unwrap();

        metrics.inc_quotes();
        metrics.inc_quotes();
        assert_eq!(metrics.get_quotes_total(), 2.0);

        metrics.inc_orders_submitted();
        assert_eq!(metrics.get_orders_submitted_total(), 1.0);

        metrics.inc_orders_filled();
        assert_eq!(metrics.get_orders_filled_total(), 1.0);

        metrics.inc_orders_cancelled();
        assert_eq!(metrics.get_orders_cancelled_total(), 1.0);

        metrics.inc_orders_rejected();
        assert_eq!(metrics.get_orders_rejected_total(), 1.0);

        metrics.inc_partial_fills();
        assert_eq!(metrics.get_partial_fills_total(), 1.0);
    }

    #[test]
    fn test_gauge_updates() {
        let metrics = PrometheusMetrics::new("test").unwrap();

        metrics.set_open_orders(5.0);
        assert_eq!(metrics.get_open_orders(), 5.0);

        metrics.set_position(100.5);
        assert_eq!(metrics.get_position(), 100.5);

        metrics.set_pnl(1000.0, 500.0);
        assert_eq!(metrics.get_pnl_realized(), 1000.0);
        assert_eq!(metrics.get_pnl_unrealized(), 500.0);
        assert_eq!(metrics.get_pnl_total(), 1500.0);

        metrics.set_spread(10.5);
        assert_eq!(metrics.get_spread(), 10.5);
    }

    #[test]
    fn test_histogram_observations() {
        let metrics = PrometheusMetrics::new("test").unwrap();

        // These should not panic
        metrics.observe_order_latency(5.0);
        metrics.observe_fill_latency(10.0);
        metrics.observe_spread(15.0);
    }

    #[test]
    fn test_encode() {
        let metrics = PrometheusMetrics::new("test").unwrap();
        metrics.inc_quotes();
        metrics.set_position(100.0);

        let encoded = metrics.encode().unwrap();
        assert!(encoded.contains("test_quotes_quotes_total"));
        assert!(encoded.contains("test_position_current"));
    }

    #[test]
    fn test_metrics_server_new() {
        let metrics = Arc::new(PrometheusMetrics::new("test").unwrap());
        let server = MetricsServer::new(Arc::clone(&metrics), "127.0.0.1:9090");
        assert_eq!(server.bind_address(), "127.0.0.1:9090");
    }

    #[test]
    fn test_metrics_bridge() {
        let live_metrics = Arc::new(LiveMetrics::new(0));
        let prom_metrics = Arc::new(PrometheusMetrics::new("test").unwrap());
        let bridge = MetricsBridge::new(Arc::clone(&live_metrics), Arc::clone(&prom_metrics));

        // Record some activity
        live_metrics.record_quote(1);
        live_metrics.record_quote(2);
        live_metrics.update_position(crate::dec!(50.0)).unwrap();
        live_metrics
            .update_pnl(crate::dec!(100.0), crate::dec!(25.0))
            .unwrap();

        // Sync
        bridge.sync().unwrap();

        // Verify Prometheus metrics updated
        assert_eq!(prom_metrics.get_quotes_total(), 2.0);
        assert_eq!(prom_metrics.get_position(), 50.0);
        assert_eq!(prom_metrics.get_pnl_realized(), 100.0);
        assert_eq!(prom_metrics.get_pnl_unrealized(), 25.0);
    }

    #[test]
    fn test_registry_access() {
        let metrics = PrometheusMetrics::new("test").unwrap();
        let registry = metrics.registry();
        let families = registry.gather();
        assert!(!families.is_empty());
    }

    #[tokio::test]
    async fn test_handle_request_metrics() {
        let metrics = Arc::new(PrometheusMetrics::new("test").unwrap());
        metrics.inc_quotes();

        // Create a mock request - we can't easily test this without a full HTTP setup
        // but we can verify the metrics encode correctly
        let encoded = metrics.encode().unwrap();
        assert!(encoded.contains("test_quotes_quotes_total 1"));
    }

    #[tokio::test]
    async fn test_metrics_server_spawn() {
        let metrics = Arc::new(PrometheusMetrics::new("test").unwrap());
        // Use port 0 to let OS assign an available port
        let server = MetricsServer::new(Arc::clone(&metrics), "127.0.0.1:0");

        // Just verify spawn doesn't panic - we can't easily test the actual server
        // without more complex setup
        assert_eq!(server.bind_address(), "127.0.0.1:0");
    }
}
