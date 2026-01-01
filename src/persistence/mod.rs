//! Persistence layer for market maker data.
//!
//! This module provides abstractions for persisting market maker data including:
//! - Trade fills and executions
//! - Position snapshots
//! - Configuration settings
//! - Daily P&L records
//! - Events and alerts
//!
//! # Example
//!
//! ```rust,ignore
//! use market_maker_rs::persistence::{Repository, InMemoryRepository, Fill};
//!
//! let repo = InMemoryRepository::new();
//!
//! // Save a fill
//! let fill = Fill::new("BTC", dec!(50000.0), dec!(1.0), FillSide::Buy, "order-1");
//! repo.save_fill(&fill).await?;
//!
//! // Get fills for a time range
//! let fills = repo.get_fills(start_time, end_time).await?;
//! ```

mod memory;
mod repository;
mod types;

pub use memory::InMemoryRepository;
pub use repository::Repository;
pub use types::{ConfigEntry, DailyPnL, EventLog, EventSeverity, Fill, FillSide, PositionSnapshot};
