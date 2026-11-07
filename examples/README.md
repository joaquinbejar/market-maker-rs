# Market Maker Examples

This directory contains comprehensive examples demonstrating all features and use cases of the `market-maker-rs` library.

## Running Examples

```bash
# Basic examples (no features required)
cargo run --example basic_quoting
cargo run --example inventory_management
cargo run --example inventory_skew
cargo run --example parameter_sensitivity
cargo run --example error_handling
cargo run --example real_time_simulation
cargo run --example config_comparison
cargo run --example full_strategy

# With serde feature (for Display/Debug formatting)
cargo run --example display_example --features serde

# Real data, not in this repository: decompressed files of the Hyperliquid capture
cargo run --release --features serde --example hyperliquid_hour -- BTC 1 10 <statuses> <diffs> <trades>
cargo run --release --features serde --example hyperliquid_validation -- BTC 1 10 <out dir> <statuses> <diffs> <trades>

# Research, on synthetic data only
cargo run --example offline_rl_support
cargo run --example consistent_time_travel
cargo run --release --features serde --example time_travel_queues -- synthetic bridges/python/experiments/m4_05/design.json
cargo run --release --example quote_duration
cargo run --release --features chain --example chain_pricing_evaluation -- <commit> <toolchain>
cargo run --release --features options,margin,multi-underlying --example option_quoting
```

## Examples Overview

### 1. `basic_quoting.rs` - Basic Quote Generation
**What it demonstrates:**
- Fundamental quote calculation using Avellaneda-Stoikov model
- Reservation price calculation
- Optimal spread calculation
- Decomposition of spread components (inventory risk + adverse selection)

**Key concepts:**
- How to use `calculate_reservation_price()`
- How to use `calculate_optimal_spread()`
- How to use `calculate_optimal_quotes()`
- Understanding spread composition

**Run:** `cargo run --example basic_quoting`

---

### 2. `inventory_management.rs` - Position Tracking
**What it demonstrates:**
- Building and managing positions (long/short)
- Calculating weighted average entry prices
- Position reduction and flattening
- Position flipping (long → short or short → long)
- Realized and unrealized PnL tracking

**Key concepts:**
- `InventoryPosition::update_fill()`
- `InventoryPosition::unrealized_pnl()`
- `PnL::add_realized()` and `PnL::set_unrealized()`
- Position lifecycle management

**Run:** `cargo run --example inventory_management`

---

### 3. `inventory_skew.rs` - Inventory Impact Analysis
**What it demonstrates:**
- How inventory position affects quote placement
- Quote "skewing" behavior
- Reservation price impact
- Comparison across different inventory levels

**Key insights:**
- Long positions → Lower quotes (incentivize selling)
- Short positions → Higher quotes (incentivize buying)
- Larger positions → More aggressive skew
- Spread remains relatively constant

**Run:** `cargo run --example inventory_skew`

---

### 4. `parameter_sensitivity.rs` - Parameter Analysis
**What it demonstrates:**
- Impact of risk aversion (gamma) on spreads
- Impact of volatility (sigma) on spreads
- Impact of order intensity (k) on spreads
- Impact of time to terminal on spreads
- Combined parameter effects

**Key insights:**
- Higher risk aversion → Wider spreads
- Higher volatility → Wider spreads
- Higher order intensity → Tighter spreads
- More time remaining → Wider spreads (more inventory risk)

**Run:** `cargo run --example parameter_sensitivity`

---

### 5. `error_handling.rs` - Validation & Error Handling
**What it demonstrates:**
- Configuration validation
- Market state validation
- Position update validation
- Numerical overflow detection
- Quote generation validation

**Key concepts:**
- Error types: `InvalidConfiguration`, `InvalidMarketState`, `InvalidPositionUpdate`, `NumericalError`, `InvalidQuoteGeneration`
- Proper error handling patterns
- Input validation

**Run:** `cargo run --example error_handling`

---

### 6. `real_time_simulation.rs` - Market Making Session
**What it demonstrates:**
- Simulated market making session
- Order flow with fills
- Market price movements
- Position management over time
- PnL evolution

**Key concepts:**
- Integrating all components
- Realistic market making workflow
- Time-based quote updates
- Dynamic position management

**Run:** `cargo run --example real_time_simulation`

---

### 7. `config_comparison.rs` - Strategy Comparison
**What it demonstrates:**
- Different strategy configurations (Conservative, Moderate, Aggressive, HFT)
- How configuration affects quote behavior
- Trade-offs between different approaches

**Configurations compared:**
- **Conservative:** Wide spreads, low risk, long time horizon
- **Moderate:** Balanced parameters
- **Aggressive:** Tight spreads, high risk tolerance, short horizon
- **High Frequency:** Very tight spreads, expects high order flow

**Run:** `cargo run --example config_comparison`

---

### 8. `full_strategy.rs` - Complete End-to-End
**What it demonstrates:**
- Complete market making workflow
- Configuration setup
- Quote generation
- Trade execution
- Position updates
- PnL tracking
- Market events handling

**Key concepts:**
- Full integration of all library features
- Realistic market making scenario
- Multi-trade session with position flips

**Run:** `cargo run --example full_strategy`

---

### 9. `display_example.rs` - Display Formatting (requires `serde` feature)
**What it demonstrates:**
- Display trait usage (compact JSON)
- Debug trait usage (pretty-printed JSON)
- Serialization capabilities

**Key concepts:**
- `Display` → Compact single-line JSON
- `Debug` → Pretty-printed multi-line JSON
- When feature `serde` is enabled

**Run:** `cargo run --example display_example --features serde`

---

### 10. `counterpart.rs` - Batch Counterpart (requires `options` feature)
**What it demonstrates:**
- `calibrate`: a surface from a quotes CSV with `calibrate_surface`, written as JSON (status, slices, report)
- `price`: European values and total Greeks on that surface for a requests CSV, one response row per request
- `identity`: the protocol, crate version and build commit as JSON

**Key concepts:**
- Decimal in and out of the CSV files, never through `f64`
- An error is a row with a class (`InvalidInput`, `Unsupported`, `Numerical`), never a number
- Exit code 2 only on a protocol failure

**Run:** `cargo run --example counterpart --features options -- identity`

---

## Learning Path

### For Beginners:
1. Start with `basic_quoting.rs` - Understand core calculations
2. Try `inventory_management.rs` - Learn position tracking
3. Explore `error_handling.rs` - See validation in action

### For Intermediate Users:
4. Study `inventory_skew.rs` - Understand inventory impact
5. Analyze `parameter_sensitivity.rs` - Learn parameter tuning
6. Review `config_comparison.rs` - Compare strategies

### For Advanced Users:
7. Run `real_time_simulation.rs` - See dynamic behavior
8. Study `full_strategy.rs` - Complete integration
9. Experiment with `display_example.rs` - Serialization features

## Testing Examples

All examples can be run to verify library functionality:

```bash
# Run all examples (bash)
for example in basic_quoting inventory_management inventory_skew \
               parameter_sensitivity error_handling real_time_simulation \
               config_comparison full_strategy; do
    echo "Running $example..."
    cargo run --example $example > /dev/null 2>&1 && echo "✓ $example" || echo "✗ $example"
done

# Run serde example
cargo run --example display_example --features serde > /dev/null 2>&1 && echo "✓ display_example" || echo "✗ display_example"
```

## Key Takeaways

- **Quote Generation:** Uses reservation price and optimal spread calculations
- **Inventory Management:** Tracks positions with weighted averages and PnL
- **Parameter Sensitivity:** Risk aversion, volatility, and order intensity all affect spreads
- **Error Handling:** Comprehensive validation at every level
- **Flexibility:** Library supports multiple strategies through configuration

## Related Documentation

- [Main README](../README.md) - Library overview
- [API Documentation](../src/lib.rs) - Detailed API docs
- [Strategy Module](../src/strategy/) - Core algorithms
- [Position Module](../src/position/) - Position tracking

## Contributing

To add a new example:

1. Create a new `.rs` file in this directory
2. Add clear documentation comments
3. Include descriptive output
4. Update this README
5. Test thoroughly
