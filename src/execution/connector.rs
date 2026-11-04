//! Exchange connector trait and core types.
//!
//! This module defines the abstract interface for exchange connectivity,
//! including order submission, cancellation, and market data retrieval.
//!
//! # Design
//!
//! The `ExchangeConnector` trait provides a unified interface for:
//! - Order lifecycle management (submit, cancel, modify)
//! - Order status queries
//! - Market data retrieval (order book snapshots)
//! - Account balance queries
//!
//! The `MarketDataStream` trait handles real-time data subscriptions.
//!
//! # Example
//!
//! ```rust
//! use market_maker_rs::execution::{
//!     OrderRequest, Action, OrderType, TimeInForce
//! };
//! use market_maker_rs::dec;
//!
//! let request = OrderRequest::new(
//!     "BTC-USD",
//!     Action::Buy,
//!     OrderType::Limit,
//!     Some(dec!(50000.0)),
//!     dec!(0.1),
//! );
//!
//! assert_eq!(request.symbol, "BTC-USD");
//! assert_eq!(request.side, Action::Buy);
//! ```

use async_trait::async_trait;
use std::fmt;

use crate::Decimal;
use crate::types::action::Dir;
use crate::types::error::{MMError, MMResult};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Unique identifier for an order.
///
/// Wraps a string identifier that uniquely identifies an order on the exchange.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::execution::OrderId;
///
/// let order_id = OrderId::new("12345");
/// assert_eq!(order_id.as_str(), "12345");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct OrderId(String);

impl OrderId {
    /// Creates a new order ID.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Returns the order ID as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consumes the OrderId and returns the inner String.
    #[must_use]
    pub fn into_inner(self) -> String {
        self.0
    }
}

impl fmt::Display for OrderId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<String> for OrderId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for OrderId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// Order direction: the shared [`financial_types::Action`].
///
/// Only [`Action::Buy`] and [`Action::Sell`] are supported; any other action
/// is rejected with [`MMError::UnsupportedAction`].
///
/// # Example
///
/// ```rust
/// use market_maker_rs::execution::Action;
///
/// let side = Action::Buy;
/// assert!(side.is_buy());
/// assert!(!side.is_sell());
/// ```
pub use crate::types::action::Action;

/// Order type.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::execution::OrderType;
///
/// let order_type = OrderType::Limit;
/// assert!(order_type.requires_price());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum OrderType {
    /// Limit order - executes at specified price or better.
    Limit,
    /// Market order - executes immediately at best available price.
    Market,
    /// Post-only order - only adds liquidity, rejected if would take.
    PostOnly,
}

impl OrderType {
    /// Returns true if this order type requires a price.
    #[must_use]
    pub fn requires_price(&self) -> bool {
        matches!(self, OrderType::Limit | OrderType::PostOnly)
    }

    /// Returns true if this is a market order.
    #[must_use]
    pub fn is_market(&self) -> bool {
        matches!(self, OrderType::Market)
    }
}

impl fmt::Display for OrderType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OrderType::Limit => write!(f, "Limit"),
            OrderType::Market => write!(f, "Market"),
            OrderType::PostOnly => write!(f, "PostOnly"),
        }
    }
}

/// Time in force - how long an order remains active.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::execution::TimeInForce;
///
/// let tif = TimeInForce::GoodTilCancel;
/// assert!(!tif.is_immediate());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Default)]
pub enum TimeInForce {
    /// Good til cancel - remains active until filled or cancelled.
    #[default]
    GoodTilCancel,
    /// Immediate or cancel - fills immediately, cancels unfilled portion.
    ImmediateOrCancel,
    /// Fill or kill - must fill entirely immediately or cancel.
    FillOrKill,
    /// Good til time - remains active until specified timestamp (milliseconds).
    GoodTilTime(u64),
}

impl TimeInForce {
    /// Returns true if this time in force requires immediate execution.
    #[must_use]
    pub fn is_immediate(&self) -> bool {
        matches!(
            self,
            TimeInForce::ImmediateOrCancel | TimeInForce::FillOrKill
        )
    }
}

impl fmt::Display for TimeInForce {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TimeInForce::GoodTilCancel => write!(f, "GTC"),
            TimeInForce::ImmediateOrCancel => write!(f, "IOC"),
            TimeInForce::FillOrKill => write!(f, "FOK"),
            TimeInForce::GoodTilTime(ts) => write!(f, "GTT({})", ts),
        }
    }
}

/// Order status with associated data.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::execution::OrderStatus;
/// use market_maker_rs::dec;
///
/// let status = OrderStatus::Open { filled_qty: dec!(0.0) };
/// assert!(status.is_open());
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum OrderStatus {
    /// Order is pending submission.
    Pending,
    /// Order is open on the book.
    Open {
        /// Quantity filled so far.
        filled_qty: Decimal,
    },
    /// Order is partially filled.
    PartiallyFilled {
        /// Quantity filled so far.
        filled_qty: Decimal,
        /// Remaining quantity.
        remaining_qty: Decimal,
    },
    /// Order is completely filled.
    Filled {
        /// Total quantity filled.
        filled_qty: Decimal,
        /// Average fill price.
        avg_price: Decimal,
    },
    /// Order was cancelled.
    Cancelled {
        /// Quantity filled before cancellation.
        filled_qty: Decimal,
    },
    /// Order was rejected.
    Rejected {
        /// Rejection reason.
        reason: String,
    },
}

impl OrderStatus {
    /// Returns true if the order is still active (pending, open, or partially filled).
    #[must_use]
    pub fn is_active(&self) -> bool {
        matches!(
            self,
            OrderStatus::Pending | OrderStatus::Open { .. } | OrderStatus::PartiallyFilled { .. }
        )
    }

    /// Returns true if the order is open on the book.
    #[must_use]
    pub fn is_open(&self) -> bool {
        matches!(
            self,
            OrderStatus::Open { .. } | OrderStatus::PartiallyFilled { .. }
        )
    }

    /// Returns true if the order is terminal (filled, cancelled, or rejected).
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            OrderStatus::Filled { .. }
                | OrderStatus::Cancelled { .. }
                | OrderStatus::Rejected { .. }
        )
    }

    /// Returns the filled quantity, if any.
    #[must_use]
    pub fn filled_qty(&self) -> Decimal {
        match self {
            OrderStatus::Pending => Decimal::ZERO,
            OrderStatus::Open { filled_qty } => *filled_qty,
            OrderStatus::PartiallyFilled { filled_qty, .. } => *filled_qty,
            OrderStatus::Filled { filled_qty, .. } => *filled_qty,
            OrderStatus::Cancelled { filled_qty } => *filled_qty,
            OrderStatus::Rejected { .. } => Decimal::ZERO,
        }
    }
}

impl fmt::Display for OrderStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OrderStatus::Pending => write!(f, "Pending"),
            OrderStatus::Open { filled_qty } => write!(f, "Open(filled={})", filled_qty),
            OrderStatus::PartiallyFilled {
                filled_qty,
                remaining_qty,
            } => write!(
                f,
                "PartiallyFilled(filled={}, remaining={})",
                filled_qty, remaining_qty
            ),
            OrderStatus::Filled {
                filled_qty,
                avg_price,
            } => {
                write!(f, "Filled(qty={}, avg_price={})", filled_qty, avg_price)
            }
            OrderStatus::Cancelled { filled_qty } => write!(f, "Cancelled(filled={})", filled_qty),
            OrderStatus::Rejected { reason } => write!(f, "Rejected({})", reason),
        }
    }
}

/// Order request for submission to an exchange.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::execution::{OrderRequest, Action, OrderType, TimeInForce};
/// use market_maker_rs::dec;
///
/// let request = OrderRequest::new(
///     "BTC-USD",
///     Action::Buy,
///     OrderType::Limit,
///     Some(dec!(50000.0)),
///     dec!(0.1),
/// );
///
/// assert_eq!(request.symbol, "BTC-USD");
/// assert_eq!(request.quantity, dec!(0.1));
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct OrderRequest {
    /// Trading symbol (e.g., "BTC-USD").
    pub symbol: String,
    /// Order side (buy or sell).
    pub side: Action,
    /// Order type (limit, market, post-only).
    pub order_type: OrderType,
    /// Limit price (required for limit and post-only orders).
    pub price: Option<Decimal>,
    /// Order quantity: in base currency, or in contracts when checked against
    /// a contract specification (`instruments::orders`).
    pub quantity: Decimal,
    /// Time in force.
    pub time_in_force: TimeInForce,
    /// Client-assigned order ID for tracking.
    pub client_order_id: Option<String>,
}

impl OrderRequest {
    /// Creates a new order request.
    ///
    /// # Arguments
    ///
    /// * `symbol` - Trading symbol
    /// * `side` - Order side
    /// * `order_type` - Order type
    /// * `price` - Limit price (None for market orders)
    /// * `quantity` - Order quantity
    #[must_use]
    pub fn new(
        symbol: impl Into<String>,
        side: Action,
        order_type: OrderType,
        price: Option<Decimal>,
        quantity: Decimal,
    ) -> Self {
        Self {
            symbol: symbol.into(),
            side,
            order_type,
            price,
            quantity,
            time_in_force: TimeInForce::default(),
            client_order_id: None,
        }
    }

    /// Creates a limit buy order.
    #[must_use]
    pub fn limit_buy(symbol: impl Into<String>, price: Decimal, quantity: Decimal) -> Self {
        Self::new(symbol, Action::Buy, OrderType::Limit, Some(price), quantity)
    }

    /// Creates a limit sell order.
    #[must_use]
    pub fn limit_sell(symbol: impl Into<String>, price: Decimal, quantity: Decimal) -> Self {
        Self::new(
            symbol,
            Action::Sell,
            OrderType::Limit,
            Some(price),
            quantity,
        )
    }

    /// Creates a market buy order.
    #[must_use]
    pub fn market_buy(symbol: impl Into<String>, quantity: Decimal) -> Self {
        Self::new(symbol, Action::Buy, OrderType::Market, None, quantity)
    }

    /// Creates a market sell order.
    #[must_use]
    pub fn market_sell(symbol: impl Into<String>, quantity: Decimal) -> Self {
        Self::new(symbol, Action::Sell, OrderType::Market, None, quantity)
    }

    /// Sets the time in force.
    #[must_use]
    pub fn with_time_in_force(mut self, tif: TimeInForce) -> Self {
        self.time_in_force = tif;
        self
    }

    /// Sets the client order ID.
    #[must_use]
    pub fn with_client_order_id(mut self, id: impl Into<String>) -> Self {
        self.client_order_id = Some(id.into());
        self
    }

    /// Returns the notional value of the order (price * quantity).
    /// Returns None for market orders without a price.
    #[must_use]
    pub fn notional(&self) -> Option<Decimal> {
        self.price.map(|p| p.saturating_mul(self.quantity))
    }
}

/// Order response from an exchange.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::execution::{OrderResponse, OrderId, OrderStatus};
/// use market_maker_rs::dec;
///
/// let response = OrderResponse {
///     order_id: OrderId::new("12345"),
///     client_order_id: Some("my-order-1".to_string()),
///     status: OrderStatus::Open { filled_qty: dec!(0.0) },
///     timestamp: 1234567890,
/// };
///
/// assert!(response.status.is_open());
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct OrderResponse {
    /// Exchange-assigned order ID.
    pub order_id: OrderId,
    /// Client-assigned order ID (if provided).
    pub client_order_id: Option<String>,
    /// Current order status.
    pub status: OrderStatus,
    /// Timestamp of the response in milliseconds.
    pub timestamp: u64,
}

impl OrderResponse {
    /// Creates a new order response.
    #[must_use]
    pub fn new(order_id: OrderId, status: OrderStatus, timestamp: u64) -> Self {
        Self {
            order_id,
            client_order_id: None,
            status,
            timestamp,
        }
    }

    /// Sets the client order ID.
    #[must_use]
    pub fn with_client_order_id(mut self, id: impl Into<String>) -> Self {
        self.client_order_id = Some(id.into());
        self
    }
}

/// Trade/fill information.
///
/// Represents a single execution (fill) of an order.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::execution::{Fill, OrderId, Action};
/// use market_maker_rs::dec;
///
/// let fill = Fill {
///     symbol: "BTC-USD".to_string(),
///     order_id: OrderId::new("12345"),
///     trade_id: "trade-1".to_string(),
///     price: dec!(50000.0),
///     quantity: dec!(0.1),
///     side: Action::Buy,
///     timestamp: 1234567890,
///     fee: dec!(0.5),
///     fee_currency: "USD".to_string(),
/// };
///
/// assert_eq!(fill.notional(), dec!(5000.0));
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Fill {
    /// Instrument traded: the symbol of the order, the identifier of a contract
    /// specification (`instruments::orders`). A record written before this
    /// field reads as empty, which no specification accepts.
    #[cfg_attr(feature = "serde", serde(default))]
    pub symbol: String,
    /// Order ID that was filled.
    pub order_id: OrderId,
    /// Unique trade/execution ID.
    pub trade_id: String,
    /// Fill price.
    pub price: Decimal,
    /// Fill quantity, in the unit of the order: base currency, or contracts of
    /// a contract specification (`instruments::orders`).
    pub quantity: Decimal,
    /// Fill side.
    pub side: Action,
    /// Fill timestamp in milliseconds.
    pub timestamp: u64,
    /// Fee amount.
    pub fee: Decimal,
    /// Fee currency.
    pub fee_currency: String,
}

impl Fill {
    /// Returns the notional value of the fill (price * quantity),
    /// saturating at the `Decimal` bounds.
    ///
    /// This is only right for a linear contract of multiplier one; on an
    /// inverse contract it is not an amount at all.
    /// `instruments::orders::fill_notional` gives the notional of a contract.
    #[must_use]
    pub fn notional(&self) -> Decimal {
        self.price.saturating_mul(self.quantity)
    }

    /// Returns the net value after fees.
    /// For buys: notional + fee (cost)
    /// For sells: notional - fee (proceeds)
    ///
    /// It is built on the same `price * quantity`, so it is only right where
    /// [`Fill::notional`] is.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::UnsupportedAction`] if the fill's side is neither
    /// buy nor sell, and [`MMError::NumericalError`]
    /// if the notional or the net value overflows.
    pub fn net_value(&self) -> MMResult<Decimal> {
        let overflow = || MMError::NumericalError("fill net value overflows".to_string());
        let notional = self.price.checked_mul(self.quantity).ok_or_else(overflow)?;
        match Dir::try_from(self.side)? {
            Dir::Buy => notional.checked_add(self.fee),
            Dir::Sell => notional.checked_sub(self.fee),
        }
        .ok_or_else(overflow)
    }
}

/// Order book price level.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::execution::BookLevel;
/// use market_maker_rs::dec;
///
/// let level = BookLevel::new(dec!(50000.0), dec!(1.5));
/// assert_eq!(level.notional(), dec!(75000.0));
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BookLevel {
    /// Price at this level.
    pub price: Decimal,
    /// Total quantity at this level.
    pub quantity: Decimal,
}

impl BookLevel {
    /// Creates a new book level.
    #[must_use]
    pub fn new(price: Decimal, quantity: Decimal) -> Self {
        Self { price, quantity }
    }

    /// Returns the notional value at this level (price * quantity),
    /// saturating at the `Decimal` bounds.
    #[must_use]
    pub fn notional(&self) -> Decimal {
        self.price.saturating_mul(self.quantity)
    }
}

/// Order book snapshot.
///
/// Contains the current state of the order book for a symbol.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::execution::{OrderBookSnapshot, BookLevel};
/// use market_maker_rs::dec;
///
/// let snapshot = OrderBookSnapshot {
///     symbol: "BTC-USD".to_string(),
///     bids: vec![
///         BookLevel::new(dec!(49990.0), dec!(1.0)),
///         BookLevel::new(dec!(49980.0), dec!(2.0)),
///     ],
///     asks: vec![
///         BookLevel::new(dec!(50010.0), dec!(1.0)),
///         BookLevel::new(dec!(50020.0), dec!(2.0)),
///     ],
///     timestamp: 1234567890,
/// };
///
/// assert_eq!(snapshot.spread(), Some(dec!(20.0)));
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct OrderBookSnapshot {
    /// Trading symbol.
    pub symbol: String,
    /// Bid levels (sorted by price descending).
    pub bids: Vec<BookLevel>,
    /// Ask levels (sorted by price ascending).
    pub asks: Vec<BookLevel>,
    /// Snapshot timestamp in milliseconds.
    pub timestamp: u64,
}

impl OrderBookSnapshot {
    /// Creates a new empty order book snapshot.
    #[must_use]
    pub fn new(symbol: impl Into<String>, timestamp: u64) -> Self {
        Self {
            symbol: symbol.into(),
            bids: Vec::new(),
            asks: Vec::new(),
            timestamp,
        }
    }

    /// Returns the best bid price.
    #[must_use]
    pub fn best_bid(&self) -> Option<Decimal> {
        self.bids.first().map(|l| l.price)
    }

    /// Returns the best ask price.
    #[must_use]
    pub fn best_ask(&self) -> Option<Decimal> {
        self.asks.first().map(|l| l.price)
    }

    /// Returns the mid price, or `None` if a side is empty or the sum of
    /// the best prices overflows.
    #[must_use]
    pub fn mid_price(&self) -> Option<Decimal> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => bid.checked_add(ask)?.checked_div(Decimal::TWO),
            _ => None,
        }
    }

    /// Returns the spread (best ask - best bid), or `None` if a side is
    /// empty or the difference overflows.
    #[must_use]
    pub fn spread(&self) -> Option<Decimal> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => ask.checked_sub(bid),
            _ => None,
        }
    }

    /// Returns the spread as a percentage of mid price.
    #[must_use]
    pub fn spread_bps(&self) -> Option<Decimal> {
        match (self.spread(), self.mid_price()) {
            (Some(spread), Some(mid)) if mid > Decimal::ZERO => {
                spread.checked_div(mid)?.checked_mul(Decimal::from(10000))
            }
            _ => None,
        }
    }

    /// Returns total bid depth (sum of all bid quantities).
    ///
    /// Report only: it saturates at the `Decimal` bounds, so it must not feed a quote, a mark or state.
    #[must_use]
    pub fn bid_depth(&self) -> Decimal {
        self.bids
            .iter()
            .fold(Decimal::ZERO, |acc, l| acc.saturating_add(l.quantity))
    }

    /// Returns total ask depth (sum of all ask quantities).
    ///
    /// Report only: it saturates at the `Decimal` bounds, so it must not feed a quote, a mark or state.
    #[must_use]
    pub fn ask_depth(&self) -> Decimal {
        self.asks
            .iter()
            .fold(Decimal::ZERO, |acc, l| acc.saturating_add(l.quantity))
    }

    /// Returns the imbalance ratio: (bid_depth - ask_depth) / (bid_depth + ask_depth).
    ///
    /// Scale invariant: when a side's depth is beyond the `Decimal`
    /// bounds, both sides are summed as quantities divided by one more than
    /// the number of levels, which keeps the ratio up to rounding. Zero when
    /// the book is empty.
    #[must_use]
    pub fn imbalance(&self) -> Decimal {
        crate::types::decimal::imbalance_of_sides(
            self.bids.iter().map(|l| l.quantity),
            self.asks.iter().map(|l| l.quantity),
        )
    }
}

/// Exchange connector trait for order management.
///
/// This trait defines the interface for interacting with an exchange,
/// including order submission, cancellation, and status queries.
///
/// # Implementation Notes
///
/// - All methods are async and return `MMResult<T>`
/// - Implementations should be `Send + Sync` for use with async runtimes
/// - Consider implementing retry logic for transient failures
#[async_trait]
pub trait ExchangeConnector: Send + Sync {
    /// Submits a new order to the exchange.
    ///
    /// # Arguments
    ///
    /// * `request` - The order request to submit
    ///
    /// # Returns
    ///
    /// The order response with the assigned order ID and initial status.
    ///
    /// # Errors
    ///
    /// Implementations return [`MMError::UnsupportedAction`] when
    /// `request.side` is neither buy nor sell, before any other effect, and
    /// their own errors otherwise.
    async fn submit_order(&self, request: OrderRequest) -> MMResult<OrderResponse>;

    /// Cancels an existing order.
    ///
    /// # Arguments
    ///
    /// * `order_id` - The ID of the order to cancel
    ///
    /// # Returns
    ///
    /// The order response with the final status.
    async fn cancel_order(&self, order_id: &OrderId) -> MMResult<OrderResponse>;

    /// Modifies an existing order (cancel and replace).
    ///
    /// # Arguments
    ///
    /// * `order_id` - The ID of the order to modify
    /// * `new_price` - New price (None to keep current)
    /// * `new_quantity` - New quantity (None to keep current)
    ///
    /// # Returns
    ///
    /// The order response for the new order.
    async fn modify_order(
        &self,
        order_id: &OrderId,
        new_price: Option<Decimal>,
        new_quantity: Option<Decimal>,
    ) -> MMResult<OrderResponse>;

    /// Gets the current status of an order.
    ///
    /// # Arguments
    ///
    /// * `order_id` - The ID of the order to query
    ///
    /// # Returns
    ///
    /// The current order response.
    async fn get_order_status(&self, order_id: &OrderId) -> MMResult<OrderResponse>;

    /// Gets all open orders for a symbol.
    ///
    /// # Arguments
    ///
    /// * `symbol` - The trading symbol
    ///
    /// # Returns
    ///
    /// A list of all open orders.
    async fn get_open_orders(&self, symbol: &str) -> MMResult<Vec<OrderResponse>>;

    /// Cancels all open orders for a symbol.
    ///
    /// # Arguments
    ///
    /// * `symbol` - The trading symbol
    ///
    /// # Returns
    ///
    /// A list of cancelled order responses.
    ///
    /// # Errors
    ///
    /// Every open order is attempted. If any cancel fails, the first error
    /// is returned after the others were attempted; the orders whose cancel
    /// failed stay open and tracked, so calling again retries them.
    async fn cancel_all_orders(&self, symbol: &str) -> MMResult<Vec<OrderResponse>>;

    /// Gets the current order book snapshot.
    ///
    /// # Arguments
    ///
    /// * `symbol` - The trading symbol
    /// * `depth` - Number of levels to retrieve
    ///
    /// # Returns
    ///
    /// The order book snapshot.
    async fn get_orderbook(&self, symbol: &str, depth: usize) -> MMResult<OrderBookSnapshot>;

    /// Gets the account balance for an asset.
    ///
    /// # Arguments
    ///
    /// * `asset` - The asset symbol (e.g., "BTC", "USD")
    ///
    /// # Returns
    ///
    /// The available balance.
    async fn get_balance(&self, asset: &str) -> MMResult<Decimal>;
}

/// Market data stream trait for real-time data.
///
/// This trait defines the interface for subscribing to and receiving
/// real-time market data updates.
#[async_trait]
pub trait MarketDataStream: Send + Sync {
    /// Subscribes to order book updates for a symbol.
    ///
    /// # Arguments
    ///
    /// * `symbol` - The trading symbol
    async fn subscribe_orderbook(&self, symbol: &str) -> MMResult<()>;

    /// Subscribes to trade stream for a symbol.
    ///
    /// # Arguments
    ///
    /// * `symbol` - The trading symbol
    async fn subscribe_trades(&self, symbol: &str) -> MMResult<()>;

    /// Gets the next order book update (blocking).
    ///
    /// # Returns
    ///
    /// The next order book snapshot.
    async fn next_orderbook_update(&self) -> MMResult<OrderBookSnapshot>;

    /// Gets the next trade (blocking).
    ///
    /// # Returns
    ///
    /// The next fill/trade.
    async fn next_trade(&self) -> MMResult<Fill>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dec;
    use crate::types::error::MMError;

    #[test]
    fn test_snapshot_overflowing_mid_and_spread_are_none() {
        let mut snapshot = OrderBookSnapshot::new("X", 0);
        snapshot.bids.push(BookLevel::new(Decimal::MAX, dec!(1)));
        snapshot.asks.push(BookLevel::new(Decimal::MAX, dec!(1)));
        assert_eq!(snapshot.mid_price(), None);
        assert_eq!(snapshot.spread_bps(), None);
        snapshot.bids[0].price = Decimal::MIN;
        assert_eq!(snapshot.spread(), None);
    }

    #[test]
    fn test_order_id() {
        let id = OrderId::new("12345");
        assert_eq!(id.as_str(), "12345");
        assert_eq!(id.to_string(), "12345");

        let id2: OrderId = "67890".into();
        assert_eq!(id2.as_str(), "67890");
    }

    #[test]
    fn test_side() {
        assert!(Action::Buy.is_buy());
        assert!(!Action::Buy.is_sell());
        assert!(Action::Sell.is_sell());
        assert!(!Action::Sell.is_buy());
    }

    #[test]
    fn test_order_type() {
        assert!(OrderType::Limit.requires_price());
        assert!(OrderType::PostOnly.requires_price());
        assert!(!OrderType::Market.requires_price());
        assert!(OrderType::Market.is_market());
    }

    #[test]
    fn test_time_in_force() {
        assert!(!TimeInForce::GoodTilCancel.is_immediate());
        assert!(TimeInForce::ImmediateOrCancel.is_immediate());
        assert!(TimeInForce::FillOrKill.is_immediate());
        assert!(!TimeInForce::GoodTilTime(1000).is_immediate());
    }

    #[test]
    fn test_order_status() {
        let pending = OrderStatus::Pending;
        assert!(pending.is_active());
        assert!(!pending.is_open());
        assert!(!pending.is_terminal());
        assert_eq!(pending.filled_qty(), Decimal::ZERO);

        let open = OrderStatus::Open {
            filled_qty: dec!(0.5),
        };
        assert!(open.is_active());
        assert!(open.is_open());
        assert!(!open.is_terminal());
        assert_eq!(open.filled_qty(), dec!(0.5));

        let filled = OrderStatus::Filled {
            filled_qty: dec!(1.0),
            avg_price: dec!(100.0),
        };
        assert!(!filled.is_active());
        assert!(!filled.is_open());
        assert!(filled.is_terminal());
        assert_eq!(filled.filled_qty(), dec!(1.0));

        let rejected = OrderStatus::Rejected {
            reason: "test".to_string(),
        };
        assert!(rejected.is_terminal());
        assert_eq!(rejected.filled_qty(), Decimal::ZERO);
    }

    #[test]
    fn test_order_request() {
        let request = OrderRequest::new(
            "BTC-USD",
            Action::Buy,
            OrderType::Limit,
            Some(dec!(50000.0)),
            dec!(0.1),
        );

        assert_eq!(request.symbol, "BTC-USD");
        assert_eq!(request.side, Action::Buy);
        assert_eq!(request.order_type, OrderType::Limit);
        assert_eq!(request.price, Some(dec!(50000.0)));
        assert_eq!(request.quantity, dec!(0.1));
        assert_eq!(request.notional(), Some(dec!(5000.0)));
    }

    #[test]
    fn test_order_request_builders() {
        let buy = OrderRequest::limit_buy("BTC-USD", dec!(50000.0), dec!(0.1));
        assert_eq!(buy.side, Action::Buy);
        assert_eq!(buy.order_type, OrderType::Limit);

        let sell = OrderRequest::limit_sell("BTC-USD", dec!(51000.0), dec!(0.1));
        assert_eq!(sell.side, Action::Sell);

        let market_buy = OrderRequest::market_buy("BTC-USD", dec!(0.1));
        assert_eq!(market_buy.order_type, OrderType::Market);
        assert_eq!(market_buy.price, None);
        assert_eq!(market_buy.notional(), None);
    }

    #[test]
    fn test_fill() {
        let fill = Fill {
            symbol: "BTC-USD".to_string(),
            order_id: OrderId::new("12345"),
            trade_id: "trade-1".to_string(),
            price: dec!(50000.0),
            quantity: dec!(0.1),
            side: Action::Buy,
            timestamp: 1000,
            fee: dec!(0.5),
            fee_currency: "USD".to_string(),
        };

        assert_eq!(fill.notional(), dec!(5000.0));
        assert_eq!(fill.net_value().unwrap(), dec!(5000.5)); // Buy: notional + fee

        let sell_fill = Fill {
            side: Action::Sell,
            ..fill.clone()
        };
        assert_eq!(sell_fill.net_value().unwrap(), dec!(4999.5)); // Sell: notional - fee

        let other_fill = Fill {
            side: Action::Other,
            ..fill.clone()
        };
        assert_eq!(
            other_fill.net_value(),
            Err(MMError::UnsupportedAction {
                action: Action::Other
            })
        );
    }

    #[test]
    fn test_fill_net_value_overflow_returns_numerical_error() {
        let fill = Fill {
            symbol: "BTC-USD".to_string(),
            order_id: OrderId::new("1"),
            trade_id: "t".to_string(),
            price: Decimal::MAX,
            quantity: dec!(2),
            side: Action::Buy,
            timestamp: 1,
            fee: Decimal::ZERO,
            fee_currency: "USD".to_string(),
        };
        assert_eq!(fill.notional(), Decimal::MAX);
        assert!(matches!(fill.net_value(), Err(MMError::NumericalError(_))));
        let fee_overflow = Fill {
            price: Decimal::MAX,
            quantity: Decimal::ONE,
            fee: Decimal::ONE,
            ..fill
        };
        assert!(matches!(
            fee_overflow.net_value(),
            Err(MMError::NumericalError(_))
        ));
    }

    #[test]
    fn test_book_level() {
        let level = BookLevel::new(dec!(50000.0), dec!(1.5));
        assert_eq!(level.notional(), dec!(75000.0));
    }

    #[test]
    fn test_order_book_snapshot() {
        let snapshot = OrderBookSnapshot {
            symbol: "BTC-USD".to_string(),
            bids: vec![
                BookLevel::new(dec!(49990.0), dec!(1.0)),
                BookLevel::new(dec!(49980.0), dec!(2.0)),
            ],
            asks: vec![
                BookLevel::new(dec!(50010.0), dec!(1.0)),
                BookLevel::new(dec!(50020.0), dec!(2.0)),
            ],
            timestamp: 1000,
        };

        assert_eq!(snapshot.best_bid(), Some(dec!(49990.0)));
        assert_eq!(snapshot.best_ask(), Some(dec!(50010.0)));
        assert_eq!(snapshot.mid_price(), Some(dec!(50000.0)));
        assert_eq!(snapshot.spread(), Some(dec!(20.0)));
        assert_eq!(snapshot.bid_depth(), dec!(3.0));
        assert_eq!(snapshot.ask_depth(), dec!(3.0));
        assert_eq!(snapshot.imbalance(), Decimal::ZERO);
    }

    #[test]
    fn test_order_book_imbalance() {
        let snapshot = OrderBookSnapshot {
            symbol: "BTC-USD".to_string(),
            bids: vec![BookLevel::new(dec!(100.0), dec!(3.0))],
            asks: vec![BookLevel::new(dec!(101.0), dec!(1.0))],
            timestamp: 1000,
        };

        // (3 - 1) / (3 + 1) = 0.5
        assert_eq!(snapshot.imbalance(), dec!(0.5));
    }

    #[test]
    fn test_empty_order_book() {
        let snapshot = OrderBookSnapshot::new("BTC-USD", 1000);

        assert_eq!(snapshot.best_bid(), None);
        assert_eq!(snapshot.best_ask(), None);
        assert_eq!(snapshot.mid_price(), None);
        assert_eq!(snapshot.spread(), None);
        assert_eq!(snapshot.imbalance(), Decimal::ZERO);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serialization() {
        let request = OrderRequest::limit_buy("BTC-USD", dec!(50000.0), dec!(0.1));
        let json = serde_json::to_string(&request).unwrap();
        let deserialized: OrderRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(request, deserialized);
    }
}
