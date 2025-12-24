# Implement Parameter Calibration Tools

## Summary

Add tools to calibrate strategy parameters from historical data for optimal performance.

## Motivation

Optimal parameter selection is crucial for strategy performance:
- Risk aversion (γ) affects inventory management aggressiveness
- Order intensity (k) should reflect actual market conditions
- Parameters may need adjustment for different market regimes

## Detailed Description

Implement calibration tools that estimate optimal parameters from historical trade and market data.

### Calibration Methods

1. **Risk Aversion (γ)**: Based on desired inventory half-life
2. **Order Intensity (k)**: From historical fill rates at different spreads
3. **Volatility Regime**: Detect and adapt to volatility regimes

### Proposed API

```rust
/// Calibration configuration
pub struct CalibrationConfig {
    /// Minimum data points required
    pub min_samples: usize,
    
    /// Confidence level for estimates
    pub confidence_level: Decimal,
    
    /// Whether to use robust estimation (outlier-resistant)
    pub robust_estimation: bool,
}

/// Calibration result
#[derive(Debug, Clone)]
pub struct CalibrationResult<T> {
    /// Estimated value
    pub value: T,
    
    /// Confidence interval (low, high)
    pub confidence_interval: (T, T),
    
    /// Number of samples used
    pub sample_size: usize,
    
    /// Quality score (0 to 1)
    pub quality: Decimal,
    
    /// Warnings or notes
    pub notes: Vec<String>,
}

/// Risk aversion calibrator
pub struct RiskAversionCalibrator {
    config: CalibrationConfig,
}

impl RiskAversionCalibrator {
    pub fn new(config: CalibrationConfig) -> Self;
    
    /// Calibrate γ based on desired inventory half-life
    ///
    /// Half-life is the time for inventory to decay to 50% through quote skewing.
    /// Formula: γ = ln(2) / (half_life * σ²)
    pub fn calibrate_from_halflife(
        &self,
        desired_halflife_ms: u64,
        volatility: Decimal,
    ) -> CalibrationResult<Decimal>;
    
    /// Calibrate γ from historical inventory and PnL data
    ///
    /// Finds γ that would have minimized inventory variance while maintaining profitability.
    pub fn calibrate_from_history(
        &self,
        inventory_history: &[(u64, Decimal)],  // (timestamp, inventory)
        pnl_history: &[(u64, Decimal)],        // (timestamp, pnl)
        volatility: Decimal,
    ) -> MMResult<CalibrationResult<Decimal>>;
}

/// Order intensity calibrator
pub struct OrderIntensityCalibrator {
    config: CalibrationConfig,
}

impl OrderIntensityCalibrator {
    pub fn new(config: CalibrationConfig) -> Self;
    
    /// Calibrate k from fill observations
    ///
    /// Uses regression on: ln(fill_rate) = ln(A) - k * spread
    pub fn calibrate_from_fills(
        &self,
        fill_observations: &[FillObservation],
    ) -> MMResult<CalibrationResult<Decimal>>;
    
    /// Calibrate k from order book data
    ///
    /// Estimates arrival rate from order book changes.
    pub fn calibrate_from_orderbook(
        &self,
        orderbook_snapshots: &[OrderBookSnapshot],
    ) -> MMResult<CalibrationResult<Decimal>>;
}

/// Volatility regime detector
pub struct VolatilityRegimeDetector {
    /// Threshold for regime change detection
    regime_threshold: Decimal,
    
    /// Lookback window in milliseconds
    lookback_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VolatilityRegime {
    Low,
    Normal,
    High,
    Extreme,
}

impl VolatilityRegimeDetector {
    pub fn new(regime_threshold: Decimal, lookback_ms: u64) -> Self;
    
    /// Detect current regime from recent volatility
    pub fn detect_regime(
        &self,
        current_volatility: Decimal,
        historical_volatility: Decimal,
    ) -> VolatilityRegime;
    
    /// Get recommended parameter adjustments for regime
    pub fn regime_adjustments(&self, regime: VolatilityRegime) -> RegimeAdjustments;
}

#[derive(Debug, Clone)]
pub struct RegimeAdjustments {
    /// Multiplier for risk aversion
    pub gamma_multiplier: Decimal,
    
    /// Multiplier for minimum spread
    pub spread_multiplier: Decimal,
    
    /// Multiplier for position limits
    pub position_limit_multiplier: Decimal,
}

/// Combined parameter optimizer
pub struct ParameterOptimizer {
    risk_aversion_calibrator: RiskAversionCalibrator,
    order_intensity_calibrator: OrderIntensityCalibrator,
    regime_detector: VolatilityRegimeDetector,
}

impl ParameterOptimizer {
    pub fn new(config: CalibrationConfig) -> Self;
    
    /// Run full calibration and return recommended parameters
    pub fn optimize(
        &self,
        market_data: &MarketDataBundle,
        current_config: &StrategyConfig,
    ) -> MMResult<OptimizedParameters>;
}

#[derive(Debug, Clone)]
pub struct OptimizedParameters {
    pub risk_aversion: CalibrationResult<Decimal>,
    pub order_intensity: CalibrationResult<Decimal>,
    pub regime: VolatilityRegime,
    pub adjustments: RegimeAdjustments,
}
```

## Acceptance Criteria

- [ ] `CalibrationConfig` and `CalibrationResult` types
- [ ] `RiskAversionCalibrator` with half-life and historical methods
- [ ] `OrderIntensityCalibrator` with fill-based estimation
- [ ] `VolatilityRegimeDetector` with regime classification
- [ ] `RegimeAdjustments` for parameter scaling
- [ ] `ParameterOptimizer` combining all calibrators
- [ ] Confidence intervals for estimates
- [ ] Quality scores indicating estimate reliability
- [ ] Unit tests with synthetic data
- [ ] Documentation with:
  - Mathematical derivations
  - Recommended parameter ranges
  - Calibration frequency guidelines
- [ ] Example showing calibration workflow

## Technical Notes

- Half-life formula derivation from A-S model dynamics
- k estimation: log-linear regression, handle zero fills carefully
- Regime thresholds: typically 0.5x, 1.5x, 2.5x of normal volatility
- Consider adding grid search optimization for complex cases
- Robust estimation: use median instead of mean, trim outliers

## Labels

`enhancement`, `optimization`, `priority:low`
