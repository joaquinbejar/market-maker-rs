# Add Prometheus Metrics Export (Feature Flag)

## Summary

Export metrics in Prometheus format for integration with Grafana dashboards and alerting.

## Motivation

Prometheus + Grafana is the industry standard for monitoring:
- Time-series storage and querying
- Beautiful dashboards
- Flexible alerting rules
- Easy integration with existing infrastructure

## Detailed Description

Add an optional `prometheus` feature that exposes metrics in Prometheus format via HTTP endpoint.

### Proposed API

```rust
// In Cargo.toml:
// [features]
// prometheus = ["dep:prometheus", "dep:hyper", "dep:tokio"]

#[cfg(feature = "prometheus")]
pub mod prometheus_export {
    use prometheus::{Registry, Counter, Gauge, Histogram, HistogramOpts};
    
    /// Prometheus metrics registry
    pub struct PrometheusMetrics {
        registry: Registry,
        
        // Counters
        quotes_total: Counter,
        orders_submitted_total: Counter,
        orders_filled_total: Counter,
        orders_cancelled_total: Counter,
        
        // Gauges
        open_orders: Gauge,
        position: Gauge,
        pnl_realized: Gauge,
        pnl_unrealized: Gauge,
        spread_current: Gauge,
        
        // Histograms
        order_latency: Histogram,
        fill_latency: Histogram,
        spread_histogram: Histogram,
    }
    
    impl PrometheusMetrics {
        /// Create new metrics registry with all metrics registered
        pub fn new(namespace: &str) -> Result<Self, prometheus::Error>;
        
        // Counter increments
        pub fn inc_quotes(&self);
        pub fn inc_orders_submitted(&self);
        pub fn inc_orders_filled(&self);
        pub fn inc_orders_cancelled(&self);
        
        // Gauge updates
        pub fn set_open_orders(&self, count: f64);
        pub fn set_position(&self, position: f64);
        pub fn set_pnl(&self, realized: f64, unrealized: f64);
        pub fn set_spread(&self, spread: f64);
        
        // Histogram observations
        pub fn observe_order_latency(&self, latency_ms: f64);
        pub fn observe_fill_latency(&self, latency_ms: f64);
        pub fn observe_spread(&self, spread: f64);
        
        /// Get registry for HTTP handler
        pub fn registry(&self) -> &Registry;
        
        /// Encode metrics to Prometheus text format
        pub fn encode(&self) -> Result<String, prometheus::Error>;
    }
    
    /// HTTP server for metrics endpoint
    pub struct MetricsServer {
        metrics: Arc<PrometheusMetrics>,
        bind_address: String,
    }
    
    impl MetricsServer {
        pub fn new(metrics: Arc<PrometheusMetrics>, bind_address: &str) -> Self;
        
        /// Start HTTP server (blocking)
        pub async fn run(&self) -> Result<(), Box<dyn std::error::Error>>;
        
        /// Start HTTP server in background
        pub fn spawn(self) -> tokio::task::JoinHandle<()>;
    }
}
```

### Example Grafana Dashboard

The feature should include a sample Grafana dashboard JSON that displays:
- PnL over time (realized + unrealized)
- Position history
- Order rates (submitted, filled, cancelled)
- Latency percentiles
- Spread distribution
- Fill rate

## Acceptance Criteria

- [ ] `prometheus` feature flag in Cargo.toml
- [ ] `PrometheusMetrics` struct with all metric types
- [ ] Counters: quotes, orders (submitted/filled/cancelled)
- [ ] Gauges: position, PnL, spread, open orders
- [ ] Histograms: latency, spread distribution
- [ ] `encode()` method for text format output
- [ ] `MetricsServer` HTTP endpoint at `/metrics`
- [ ] Proper metric naming (namespace_subsystem_name)
- [ ] Metric labels where appropriate (e.g., order side)
- [ ] Sample Grafana dashboard JSON in `doc/grafana/`
- [ ] Integration with `LiveMetrics` (bridge adapter)
- [ ] Unit tests for metric registration and encoding
- [ ] Documentation with setup instructions
- [ ] Example showing full integration

## Technical Notes

- Use `prometheus` crate for metric types
- Use `hyper` or `warp` for HTTP server
- Metric naming convention: `marketmaker_<subsystem>_<name>_<unit>`
- Example: `marketmaker_orders_filled_total`, `marketmaker_latency_milliseconds`
- Histogram buckets for latency: [0.1, 0.5, 1, 2, 5, 10, 25, 50, 100, 250, 500, 1000]
- Default port: 9090 (configurable)

## Labels

`enhancement`, `monitoring`, `priority:low`
