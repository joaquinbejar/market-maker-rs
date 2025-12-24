# Implement Correlation Matrix and Portfolio Risk

## Summary

Support multi-asset portfolios with correlation-aware risk management.

## Motivation

Market makers often trade multiple correlated assets:
- Cross-margining benefits from correlation
- Hedging opportunities between correlated pairs
- Portfolio-level risk limits more accurate than per-asset limits
- Spread trading between correlated assets

## Detailed Description

Implement correlation tracking and portfolio-level risk calculations.

### Proposed API

```rust
/// Asset identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AssetId(pub String);

/// Correlation matrix
#[derive(Debug, Clone)]
pub struct CorrelationMatrix {
    assets: Vec<AssetId>,
    correlations: Vec<Decimal>,  // Symmetric matrix as flat vector
    last_update: u64,
}

impl CorrelationMatrix {
    pub fn new(assets: Vec<AssetId>) -> Self;
    pub fn identity(assets: Vec<AssetId>) -> Self;
    pub fn get_correlation(&self, asset1: &AssetId, asset2: &AssetId) -> Option<Decimal>;
    pub fn set_correlation(&mut self, asset1: &AssetId, asset2: &AssetId, correlation: Decimal) -> MMResult<()>;
    pub fn update_from_returns(&mut self, returns: &HashMap<AssetId, Vec<Decimal>>, timestamp: u64) -> MMResult<()>;
    pub fn is_valid(&self) -> bool;
}

/// Portfolio position
#[derive(Debug, Clone)]
pub struct PortfolioPosition {
    positions: HashMap<AssetId, Decimal>,
    volatilities: HashMap<AssetId, Decimal>,
}

/// Portfolio risk calculator
pub struct PortfolioRiskCalculator {
    correlation_matrix: CorrelationMatrix,
}

impl PortfolioRiskCalculator {
    pub fn new(correlation_matrix: CorrelationMatrix) -> Self;
    pub fn portfolio_variance(&self, portfolio: &PortfolioPosition) -> MMResult<Decimal>;
    pub fn portfolio_volatility(&self, portfolio: &PortfolioPosition) -> MMResult<Decimal>;
    pub fn portfolio_var(&self, portfolio: &PortfolioPosition, confidence: Decimal, horizon_days: u32) -> MMResult<Decimal>;
    pub fn marginal_risk_contribution(&self, portfolio: &PortfolioPosition) -> MMResult<HashMap<AssetId, Decimal>>;
}

/// Cross-asset hedging calculator
pub struct HedgeCalculator {
    correlation_matrix: CorrelationMatrix,
}

impl HedgeCalculator {
    pub fn hedge_ratio(&self, target: &AssetId, hedge: &AssetId, target_vol: Decimal, hedge_vol: Decimal) -> MMResult<Decimal>;
    pub fn find_best_hedge(&self, target: &AssetId, available: &[AssetId]) -> Option<(AssetId, Decimal)>;
}
```

## Acceptance Criteria

- [ ] `AssetId` type for asset identification
- [ ] `CorrelationMatrix` with get/set operations
- [ ] `update_from_returns()` calculates correlations from data
- [ ] Matrix validation (symmetric, proper range [-1, 1])
- [ ] `PortfolioPosition` tracking multi-asset positions
- [ ] `PortfolioRiskCalculator` with variance/volatility/VaR
- [ ] `marginal_risk_contribution()` for risk attribution
- [ ] `HedgeCalculator` for cross-asset hedging
- [ ] Unit tests for all calculations
- [ ] Documentation with mathematical formulas

## Technical Notes

- Portfolio variance: `σ²_p = Σᵢ Σⱼ wᵢ wⱼ σᵢ σⱼ ρᵢⱼ`
- Hedge ratio: `β = ρ * (σ_target / σ_hedge)`
- VaR (parametric): `VaR = -μ + z * σ * sqrt(horizon)`
- Store only upper triangle of correlation matrix

## Labels

`enhancement`, `multi-asset`, `risk`, `priority:low`
