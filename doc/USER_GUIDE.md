# Market Maker RS - User Guide

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Getting Started](#getting-started)
- [Core Concepts](#core-concepts)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Architecture](#architecture)
- [Testing](#testing)
- [Performance](#performance)
- [Contributing](#contributing)

---

## Overview

**Market Maker RS** is a high-performance Rust library implementing quantitative market-making strategies, with a focus on the Avellaneda-Stoikov model. This library provides tools for calculating optimal bid-ask spreads, managing inventory risk, and estimating market volatility with precision using arbitrary-precision decimal arithmetic.

### What is Market Making?

Market making is a trading strategy where a participant (the market maker) continuously provides liquidity to a market by quoting both bid and ask prices. The goal is to profit from the bid-ask spread while managing inventory risk and adverse selection.

### Avellaneda-Stoikov Model

The Avellaneda-Stoikov model is a continuous-time model for optimal market making that balances:
- **Inventory risk**: Risk from holding positions
- **Adverse selection**: Risk of trading with informed traders
- **Profit maximization**: Capturing the bid-ask spread

**Key Formula:**
```
δ = γ * σ² * (T - t) + (2/γ) * ln(1 + γ/k)
```

Where:
- `δ` = optimal half-spread
- `γ` = risk aversion parameter
- `σ` = volatility
- `T - t` = time to terminal
- `k` = order arrival intensity

---

## Key Features

### ✅ Core Functionality

- **Avellaneda-Stoikov Strategy Implementation**
  - Reservation price calculation
  - Optimal spread computation
  - Dynamic bid-ask quote generation

- **Precision Arithmetic**
  - Uses `rust_decimal` for financial calculations
  - Eliminates floating-point rounding errors
  - Exact decimal representation

- **Volatility Estimation**
  - Simple historical volatility
  - EWMA (Exponentially Weighted Moving Average)
  - Parkinson's range-based estimator

- **Position Management**
  - Inventory tracking
  - PnL (Profit & Loss) calculation
  - Position risk metrics

- **Flexible Architecture**
  - Synchronous and asynchronous trait interfaces
  - Extensible strategy patterns
  - Zero-cost abstractions

### 🔧 Advanced Features

- **Custom Volatility Estimators**: Implement your own volatility models
- **Strategy Composition**: Combine multiple strategies
- **Real-time Adaptation**: Dynamic parameter adjustment
- **Risk Controls**: Built-in validation and error handling

---

## Technology Stack

### Core Dependencies

| Technology | Version | Purpose |
|------------|---------|---------|
| **Rust** | 2024 Edition | Core language |
| **rust_decimal** | 1.36 | Arbitrary-precision decimal arithmetic |
| **thiserror** | 2.0 | Error handling |
| **async-trait** | 0.1 | Async trait support |
| **serde** | 1.0 | Serialization (optional) |
| **tokio** | 1.0 | Async runtime (dev) |

### Why Rust?

1. **Performance**: Zero-cost abstractions, no garbage collector
2. **Safety**: Memory safety without runtime overhead
3. **Concurrency**: Fearless concurrency with ownership model
4. **Reliability**: Strong type system catches errors at compile time

### Why Decimal Arithmetic?

Financial calculations require exact precision. Floating-point arithmetic (`f64`) can introduce rounding errors:

```rust
// ❌ Floating point error
let price = 0.1 + 0.2;  // 0.30000000000000004

// ✅ Exact decimal arithmetic
let price = dec!(0.1) + dec!(0.2);  // 0.3
```

---

## Getting Started

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
market-maker-rs = "0.1.2"
```

With optional features:

```toml
[dependencies]
market-maker-rs = { version = "0.1.2", features = ["serde"] }
```

### Quick Start

```rust
use market_maker_rs::prelude::*;
use market_maker_rs::strategy::avellaneda_stoikov::calculate_optimal_quotes;

fn main() {
    // Market parameters
    let mid_price = dec!(100.0);
    let inventory = dec!(0.0);  // Flat position
    let risk_aversion = dec!(0.1);
    let volatility = dec!(0.2);  // 20% annualized
    let time_to_terminal = 3600000;  // 1 hour in ms
    let order_intensity = dec!(1.5);

    // Calculate optimal quotes
    let (bid, ask) = calculate_optimal_quotes(
        mid_price,
        inventory,
        risk_aversion,
        volatility,
        time_to_terminal,
        order_intensity,
    ).expect("Failed to calculate quotes");

    println!("Optimal Bid: ${:.2}", bid);
    println!("Optimal Ask: ${:.2}", ask);
    println!("Spread: ${:.4}", ask - bid);
}
```

**Output:**
```
Optimal Bid: $99.35
Optimal Ask: $100.64
Spread: $1.2907
```

---

## Core Concepts

### 1. Reservation Price

The **reservation price** is the "fair value" adjusted for inventory risk:

```rust
use market_maker_rs::strategy::avellaneda_stoikov::calculate_reservation_price;

let reservation = calculate_reservation_price(
    mid_price,
    inventory,  // Positive = long, Negative = short
    risk_aversion,
    volatility,
    time_to_terminal,
)?;
```

**Behavior:**
- **Long position** (inventory > 0): `reservation < mid_price` → Incentivize selling
- **Flat position** (inventory = 0): `reservation = mid_price`
- **Short position** (inventory < 0): `reservation > mid_price` → Incentivize buying

### 2. Optimal Spread

The **optimal spread** balances profit and risk:

```rust
use market_maker_rs::strategy::avellaneda_stoikov::calculate_optimal_spread;

let spread = calculate_optimal_spread(
    risk_aversion,
    volatility,
    time_to_terminal,
    order_intensity,
)?;
```

**Factors:**
- Higher volatility → Wider spreads
- Higher risk aversion → Wider spreads
- Higher order intensity → Tighter spreads
- Less time remaining → Wider spreads

### 3. Volatility Estimation

Accurate volatility is critical for the model:

```rust
use market_maker_rs::market_state::volatility::VolatilityEstimator;

let estimator = VolatilityEstimator::new();
let prices = vec![dec!(100.0), dec!(101.0), dec!(99.5), dec!(100.5)];

// Method 1: Simple historical volatility
let vol_simple = estimator.calculate_simple(&prices)?;

// Method 2: EWMA (more weight to recent data)
let vol_ewma = estimator.calculate_ewma(&prices, dec!(0.94))?;

// Method 3: Parkinson (uses high-low range)
let highs = vec![dec!(102.0), dec!(103.0), dec!(101.0)];
let lows = vec![dec!(99.0), dec!(100.0), dec!(98.0)];
let vol_parkinson = estimator.calculate_parkinson(&highs, &lows)?;
```

### 4. Position Management

Track inventory and PnL:

```rust
use market_maker_rs::position::{InventoryPosition, PnL};

let mut inventory = InventoryPosition::new();
let mut pnl = PnL::new();

// Execute buy
inventory.update_fill(dec!(10.0), dec!(100.5), timestamp);

// Update PnL
pnl.set_unrealized(inventory.unrealized_pnl(current_mid_price));

// Execute sell
let realized = inventory.quantity * (sell_price - inventory.avg_entry_price);
inventory.update_fill(dec!(-10.0), sell_price, timestamp);
pnl.add_realized(realized);
```

---

## API Reference

### Main Functions

#### `calculate_optimal_quotes()`

```rust
pub fn calculate_optimal_quotes(
    mid_price: Decimal,
    inventory: Decimal,
    risk_aversion: Decimal,
    volatility: Decimal,
    time_to_terminal_ms: u64,
    order_intensity: Decimal,
) -> MMResult<(Decimal, Decimal)>
```

**Returns:** `(bid_price, ask_price)`

**Errors:**
- `InvalidMarketState`: Invalid prices or volatility
- `InvalidConfiguration`: Invalid parameters (negative risk aversion, etc.)
- `InvalidQuoteGeneration`: Resulting quotes are invalid (bid >= ask, bid <= 0)

#### `calculate_reservation_price()`

```rust
pub fn calculate_reservation_price(
    mid_price: Decimal,
    inventory: Decimal,
    risk_aversion: Decimal,
    volatility: Decimal,
    time_to_terminal_ms: u64,
) -> MMResult<Decimal>
```

**Returns:** Reservation price adjusted for inventory risk

#### `calculate_optimal_spread()`

```rust
pub fn calculate_optimal_spread(
    risk_aversion: Decimal,
    volatility: Decimal,
    time_to_terminal_ms: u64,
    order_intensity: Decimal,
) -> MMResult<Decimal>
```

**Returns:** Optimal spread (full spread, not half-spread)

### Traits

#### `AvellanedaStoikov` (Synchronous)

```rust
pub trait AvellanedaStoikov {
    fn calculate_reservation_price(...) -> MMResult<Decimal>;
    fn calculate_optimal_spread(...) -> MMResult<Decimal>;
    fn calculate_optimal_quotes(...) -> MMResult<(Decimal, Decimal)>;
}
```

#### `AsyncAvellanedaStoikov` (Asynchronous)

```rust
#[async_trait]
pub trait AsyncAvellanedaStoikov {
    async fn calculate_reservation_price(...) -> MMResult<Decimal>;
    async fn calculate_optimal_spread(...) -> MMResult<Decimal>;
    async fn calculate_optimal_quotes(...) -> MMResult<(Decimal, Decimal)>;
}
```

### Configuration

#### `StrategyConfig`

```rust
pub struct StrategyConfig {
    pub risk_aversion: Decimal,      // γ (gamma)
    pub order_intensity: Decimal,    // k
    pub terminal_time: u64,          // T (milliseconds)
    pub min_spread: Decimal,         // Minimum spread override
}

impl StrategyConfig {
    pub fn new(
        risk_aversion: Decimal,
        order_intensity: Decimal,
        terminal_time: u64,
        min_spread: Decimal,
    ) -> MMResult<Self>;
}
```

**Validation:**
- `risk_aversion` > 0
- `order_intensity` > 0
- `min_spread` >= 0

---

## Examples

### Example 1: Basic Quote Generation

See: `examples/basic_quoting.rs`

```bash
cargo run --example basic_quoting
```

Demonstrates fundamental quote calculation with different inventory levels.

### Example 2: Custom Strategy with Traits

See: `examples/trait_sync_example.rs`

```bash
cargo run --example trait_sync_example
```

Shows how to implement a custom strategy by extending the base implementation.

### Example 3: Async Strategy with External Data

See: `examples/trait_async_example.rs`

```bash
cargo run --example trait_async_example
```

Demonstrates async trait usage with simulated external API calls for real-time data.

### Example 4: Volatility Estimation

See: `examples/volatility_estimation.rs`

```bash
cargo run --example volatility_estimation
```

Compares different volatility estimation methods and their impact on spreads.

### Example 5: Full Strategy Simulation

See: `examples/full_strategy.rs`

```bash
cargo run --example full_strategy
```

Complete market-making session with trades, inventory management, and PnL tracking.

### All Examples

Run all examples:

```bash
cargo run --example basic_quoting
cargo run --example config_comparison
cargo run --example display_example --features serde
cargo run --example error_handling
cargo run --example full_strategy
cargo run --example inventory_management
cargo run --example inventory_skew
cargo run --example parameter_sensitivity
cargo run --example real_time_simulation
cargo run --example trait_sync_example
cargo run --example trait_async_example
cargo run --example volatility_estimation
```

---

## Architecture

### Module Structure

```
market-maker-rs/
├── src/
│   ├── lib.rs                      # Main library entry
│   ├── market_state/
│   │   ├── mod.rs
│   │   ├── snapshot.rs             # Market state snapshots
│   │   └── volatility.rs           # Volatility estimators
│   ├── position/
│   │   ├── mod.rs
│   │   ├── inventory.rs            # Inventory tracking
│   │   └── pnl.rs                  # PnL calculation
│   ├── strategy/
│   │   ├── mod.rs
│   │   ├── avellaneda_stoikov.rs   # Core strategy implementation
│   │   ├── config.rs               # Strategy configuration
│   │   ├── interface.rs            # Trait definitions
│   │   └── quote.rs                # Quote representation
│   └── types/
│       ├── mod.rs
│       ├── decimal.rs              # Decimal math helpers
│       └── error.rs                # Error types
├── examples/                        # 12 complete examples
├── tests/                           # Integration tests
└── doc/                            # Documentation
```

### Design Patterns

1. **Trait-Based**: Extensible strategy interface
2. **Zero-Copy**: Efficient data structures
3. **Error Handling**: Result types with descriptive errors
4. **Type Safety**: Strong typing prevents runtime errors

### Data Flow

```
Market Data → Volatility Estimator → Volatility
                    ↓
Parameters → Avellaneda-Stoikov Model → Quotes
                    ↓
Quotes → Order Execution → Position Updates
                    ↓
Position → PnL Calculation → Performance Metrics
```

---

## Testing

### Run Tests

```bash
# Run all tests
cargo test

# Run library tests only
cargo test --lib

# Run tests with output
cargo test -- --nocapture

# Run specific test
cargo test test_optimal_quotes_valid
```

### Test Coverage

Current coverage: **~95%**

```bash
# Generate coverage report (requires tarpaulin)
cargo tarpaulin --out Html
```

### Test Structure

- **Unit Tests**: 91 tests covering all modules
- **Integration Tests**: 23 tests for end-to-end scenarios
- **Doc Tests**: Embedded in documentation

### Quality Assurance

```bash
# Lint code
make lint

# Fix linting issues
make lint-fix

# Run all checks (format, test, lint, doc)
make pre-push
```

---

## Performance

### Benchmarks

Typical performance on modern hardware:

| Operation | Time | Operations/sec |
|-----------|------|----------------|
| Calculate quotes | ~25 µs | ~40,000 |
| Calculate reservation price | ~10 µs | ~100,000 |
| Calculate spread | ~15 µs | ~66,000 |
| EWMA volatility (100 prices) | ~200 µs | ~5,000 |

### Optimization Tips

1. **Reuse Estimators**: Create once, use many times
2. **Batch Operations**: Calculate multiple quotes together
3. **Async for I/O**: Use async traits for external data sources
4. **Profile First**: Use `cargo bench` before optimizing

---

## Contributing

### Development Setup

```bash
# Clone repository
git clone https://github.com/joaquinbejar/market-maker-rs
cd market-maker-rs

# Run tests
cargo test

# Check code
make lint

# Build documentation
cargo doc --open

# Run all quality checks
make pre-push
```

### Code Style

- Follow Rust API Guidelines
- Document all public APIs
- Include examples in documentation
- Write tests for new features
- Use `cargo fmt` for formatting

### Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run `make pre-push`
6. Submit PR with description

---

## FAQ

### Q: Why use Decimal instead of f64?

**A:** Financial calculations require exact precision. Floating-point arithmetic introduces rounding errors that can accumulate in trading systems.

### Q: How do I choose parameters?

**A:** Parameters depend on your market and risk tolerance:
- **Risk aversion (γ)**: 0.01-1.0 (lower = more aggressive)
- **Order intensity (k)**: 0.5-5.0 (depends on market liquidity)
- **Volatility (σ)**: Use historical estimation
- **Min spread**: Set based on exchange fees

### Q: Can I use this in production?

**A:** The library provides the mathematical models. For production use, you need to add:
- Exchange connectivity
- Order management
- Risk controls
- Monitoring and alerting
- Backtesting framework

### Q: How do I handle market regimes?

**A:** Implement custom strategies using the trait interfaces to adapt parameters dynamically based on market conditions.

### Q: What about latency?

**A:** Calculations are microsecond-level. Network latency to exchanges will dominate. Use async traits for non-blocking operations.

---

## Resources

### Academic Papers

- Avellaneda, M., & Stoikov, S. (2008). "High-frequency trading in a limit order book"
- Guéant, O., Lehalle, C. A., & Fernandez-Tapia, J. (2013). "Dealing with the inventory risk"

### Documentation

- [API Documentation](https://docs.rs/market-maker-rs)
- [GitHub Repository](https://github.com/joaquinbejar/market-maker-rs)
- [Examples](../examples/)

### Community

- Issues: [GitHub Issues](https://github.com/joaquinbejar/market-maker-rs/issues)
- Discussions: [GitHub Discussions](https://github.com/joaquinbejar/market-maker-rs/discussions)

---

## License

MIT License - See [LICENSE](../LICENSE) file for details

---

## Changelog

### Version 0.1.2 (Current)

- ✅ Complete Decimal migration
- ✅ Volatility estimator implementation
- ✅ Async trait support
- ✅ 12 comprehensive examples
- ✅ 95% test coverage
- ✅ Full documentation

---

**Made with ❤️ by [Joaquin Bejar](https://github.com/joaquinbejar)**
