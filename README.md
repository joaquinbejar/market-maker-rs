[![Dual License](https://img.shields.io/badge/license-MIT%20and%20Apache%202.0-blue)](./LICENSE)
[![Crates.io](https://img.shields.io/crates/v/market-maker-rs.svg)](https://crates.io/crates/market-maker-rs)
[![Downloads](https://img.shields.io/crates/d/market-maker-rs.svg)](https://crates.io/crates/market-maker-rs)
[![Stars](https://img.shields.io/github/stars/joaquinbejar/market-maker-rs.svg)](https://github.com/joaquinbejar/market-maker-rs/stargazers)
[![Issues](https://img.shields.io/github/issues/joaquinbejar/market-maker-rs.svg)](https://github.com/joaquinbejar/market-maker-rs/issues)
[![PRs](https://img.shields.io/github/issues-pr/joaquinbejar/market-maker-rs.svg)](https://github.com/joaquinbejar/market-maker-rs/pulls)

[![Build Status](https://img.shields.io/github/workflow/status/joaquinbejar/market-maker-rs/CI)](https://github.com/joaquinbejar/market-maker-rs/actions)
[![Coverage](https://img.shields.io/codecov/c/github/joaquinbejar/market-maker-rs)](https://codecov.io/gh/joaquinbejar/market-maker-rs)
[![Dependencies](https://img.shields.io/librariesio/github/joaquinbejar/market-maker-rs)](https://libraries.io/github/joaquinbejar/market-maker-rs)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://docs.rs/market-maker-rs)



Market Making Library

A Rust library implementing quantitative market making strategies, starting with the
Avellaneda-Stoikov model. This library provides the mathematical foundations and domain models
necessary for building automated market making systems for financial markets.

## Overview

Market making is the practice of simultaneously providing buy (bid) and sell (ask) quotes
in a financial market. The market maker profits from the bid-ask spread while providing
liquidity to the market.

### Key Challenges

- **Inventory Risk**: Holding positions exposes the market maker to price movements
- **Adverse Selection**: Informed traders may trade against you when they have better information
- **Optimal Pricing**: Balance between execution probability and profitability

## The Avellaneda-Stoikov Model

The Avellaneda-Stoikov model (2008) solves the optimal market making problem using
stochastic control theory. It determines optimal bid and ask prices given:

- Current market price and volatility
- Current inventory position
- Risk aversion
- Time remaining in trading session
- Order arrival dynamics

## Modules

- [`strategy`]: Pure mathematical calculations for quote generation
- [`position`]: Inventory tracking and PnL management
- [`market_state`]: Market data representation
- [`risk`]: Position limits and exposure control
- [`types`]: Common types and error definitions
- [`prelude`]: Convenient re-exports of commonly used types

## Quick Start

Import commonly used types with the prelude:

```rust
use market_maker_rs::prelude::*;
```

## Examples

Examples will be added once core functionality is implemented.

## 🛠 Makefile Commands

This project includes a `Makefile` with common tasks to simplify development. Here's a list of useful commands:

### 🔧 Build & Run

```sh
make build         # Compile the project
make release       # Build in release mode
make run           # Run the main binary
```

### 🧪 Test & Quality

```sh
make test          # Run all tests
make fmt           # Format code
make fmt-check     # Check formatting without applying
make lint          # Run clippy with warnings as errors
make lint-fix      # Auto-fix lint issues
make fix           # Auto-fix Rust compiler suggestions
make check         # Run fmt-check + lint + test
```

### 📦 Packaging & Docs

```sh
make doc           # Check for missing docs via clippy
make doc-open      # Build and open Rust documentation
make create-doc    # Generate internal docs
make readme        # Regenerate README using cargo-readme
make publish       # Prepare and publish crate to crates.io
```

### 📈 Coverage & Benchmarks

```sh
make coverage            # Generate code coverage report (XML)
make coverage-html       # Generate HTML coverage report
make open-coverage       # Open HTML report
make bench               # Run benchmarks using Criterion
make bench-show          # Open benchmark report
make bench-save          # Save benchmark history snapshot
make bench-compare       # Compare benchmark runs
make bench-json          # Output benchmarks in JSON
make bench-clean         # Remove benchmark data
```

### 🧪 Git & Workflow Helpers

```sh
make git-log             # Show commits on current branch vs main
make check-spanish       # Check for Spanish words in code
make zip                 # Create zip without target/ and temp files
make tree                # Visualize project tree (excludes common clutter)
```

### 🤖 GitHub Actions (via act)

```sh
make workflow-build      # Simulate build workflow
make workflow-lint       # Simulate lint workflow
make workflow-test       # Simulate test workflow
make workflow-coverage   # Simulate coverage workflow
make workflow            # Run all workflows
```

ℹ️ Requires act for local workflow simulation and cargo-tarpaulin for coverage.

## Contribution and Contact

We welcome contributions to this project! If you would like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and ensure that the project still builds and all tests pass.
4. Commit your changes and push your branch to your forked repository.
5. Submit a pull request to the main repository.

If you have any questions, issues, or would like to provide feedback, please feel free to contact the project
maintainer:

### **Contact Information**
- **Author**: Joaquín Béjar García
- **Email**: jb@taunais.com
- **Telegram**: [@joaquin_bejar](https://t.me/joaquin_bejar)
- **Repository**: <https://github.com/joaquinbejar/market-maker-rs>
- **Documentation**: <https://docs.rs/market-maker-rs>


We appreciate your interest and look forward to your contributions!

**License**: MIT
