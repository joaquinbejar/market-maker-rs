//! Prelude module for convenient imports.
//!
//! This module re-exports the most commonly used types, traits, and functions
//! from the market making library. Users can import everything they need with:
//!
//! ```rust
//! use market_maker_rs::prelude::*;
//! ```

// Re-export Decimal and helper functions
pub use crate::types::decimal::{decimal_ln, decimal_powi, decimal_sqrt};
pub use crate::{Decimal, dec};

// Re-export types module
pub use crate::types::error::{MMError, MMResult};
pub use crate::types::primitives::{
    OrderIntensity, Price, Quantity, RiskAversion, Timestamp, Volatility,
};

// Re-export strategy types
pub use crate::strategy::adaptive_spread::{
    AdaptiveSpread, AdaptiveSpreadCalculator, AdaptiveSpreadConfig, OrderBookImbalance, Trade,
    TradeFlowImbalance,
};
pub use crate::strategy::calibration::{
    CalibrationConfig, CalibrationResult, FillObservation as CalibrationFillObservation,
    OptimizedParameters, OrderIntensityCalibrator, ParameterOptimizer, RegimeAdjustments,
    RiskAversionCalibrator, VolatilityRegime, VolatilityRegimeDetector,
};
pub use crate::strategy::config::StrategyConfig;
pub use crate::strategy::glft::{GLFTConfig, GLFTStrategy, PenaltyFunction};
pub use crate::strategy::grid::{GridConfig, GridOrder, GridStrategy, OrderSide};
pub use crate::strategy::quote::Quote;

// Re-export position types
pub use crate::position::inventory::InventoryPosition;
pub use crate::position::pnl::PnL;

// Re-export market state types
pub use crate::market_state::snapshot::MarketState;

// Re-export risk types
pub use crate::risk::{
    Alert, AlertHandler, AlertManager, AlertSeverity, AlertType, AssetId, CallbackAlertHandler,
    CircuitBreaker, CircuitBreakerConfig, CircuitBreakerState, CollectingAlertHandler,
    CorrelationMatrix, DrawdownRecord, DrawdownTracker, HedgeCalculator, LogAlertHandler,
    PortfolioPosition, PortfolioRiskCalculator, RiskLimits, TriggerReason,
};

// Re-export analytics types
pub use crate::analytics::intensity::{
    FillObservation, FillSide, IntensityEstimate, ObservationStats, OrderIntensityConfig,
    OrderIntensityEstimator,
};
pub use crate::analytics::live_metrics::{
    Counter, Gauge, LiveMetrics, MetricsSnapshot, SharedLiveMetrics,
};
pub use crate::analytics::order_flow::{
    OrderFlowAnalyzer, OrderFlowAnalyzerBuilder, OrderFlowStats, TradeSide,
};
#[cfg(feature = "prometheus")]
pub use crate::analytics::prometheus_export::{MetricsBridge, MetricsServer, PrometheusMetrics};
pub use crate::analytics::vpin::{
    BucketStats, TradeClassifier, VPINCalculator, VPINConfig, VolumeBucket,
};

// Re-export execution types
pub use crate::execution::{
    BookLevel, ExchangeConnector, Fill, Histogram, LatencyMeasurement, LatencyMetric, LatencyStats,
    LatencyTracker, LatencyTrackerConfig, ManagedOrder, MarketDataStream, MockConfig,
    MockExchangeConnector, OrderBookConnector, OrderBookConnectorConfig, OrderBookSnapshot,
    OrderId, OrderManager, OrderManagerConfig, OrderManagerStats, OrderRequest, OrderResponse,
    OrderStatus, OrderType, Side, ThreadSafeOrderManager, TimeInForce,
};

// Re-export backtest types
pub use crate::backtest::{
    BacktestConfig, BacktestEngine, BacktestResult, BacktestStrategy, EquityPoint, FillModel,
    FillResult, HistoricalDataSource, ImmediateFillModel, MarketImpactFillModel, MarketTick,
    MetricsCalculator, MetricsConfig, OHLCVBar, PerformanceMetrics, ProbabilisticFillModel,
    QueuePositionFillModel, SimulatedFill, SimulatedOrder, SlippageModel, TradeRecord,
    VecDataSource,
};

// Re-export options types (when feature is enabled)
#[cfg(feature = "options")]
pub use crate::options::{
    GreeksLimits, HedgeOrder, HedgeType, OptionsAdapter, OptionsMarketMaker,
    OptionsMarketMakerConfig, OptionsMarketMakerImpl, PortfolioGreeks, PositionGreeks,
};
