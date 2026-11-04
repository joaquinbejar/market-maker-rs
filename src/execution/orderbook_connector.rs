//! OrderBook-rs connector implementation.
//!
//! This module provides an `ExchangeConnector` implementation that connects
//! to the `orderbook-rs` library for order management and execution.
//!
//! # Example
//!
//! ```rust,ignore
//! use market_maker_rs::execution::{OrderBookConnector, OrderRequest, Action, OrderType};
//! use market_maker_rs::dec;
//!
//! // Create connector with an order book
//! let connector = OrderBookConnector::new("BTC-USD");
//!
//! // Submit a limit order
//! let request = OrderRequest::limit_buy("BTC-USD", dec!(50000.0), dec!(1.0));
//! let response = connector.submit_order(request).await?;
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use orderbook_rs::{
    DefaultOrderBook, OrderBook, OrderBookError, OrderBookSnapshot as OBSnapshot,
    OrderId as OBOrderId, Side as OBSide, TimeInForce as OBTimeInForce,
};

use crate::Decimal;
use crate::execution::connector::{
    BookLevel, ExchangeConnector, OrderBookSnapshot, OrderId, OrderRequest, OrderResponse,
    OrderStatus, OrderType, TimeInForce,
};
use crate::types::action::{Action, Dir};
use crate::types::error::{MMError, MMResult};
use rust_decimal::prelude::FromPrimitive;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Configuration for the OrderBook connector.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct OrderBookConnectorConfig {
    /// Price precision (decimal places).
    pub price_precision: u32,
    /// Quantity precision (decimal places).
    pub quantity_precision: u32,
    /// Fee rate as a decimal (e.g., 0.001 for 0.1%).
    pub fee_rate: Decimal,
    /// Fee currency.
    pub fee_currency: String,
}

/// Largest supported price or quantity precision: `10^19` is the largest
/// power of ten that fits in a `u64`.
pub const MAX_PRECISION: u32 = 19;

/// Returns `10^exponent` as a `u128`; `None` when it does not fit
/// (exponent above 38).
fn pow10(exponent: u32) -> Option<u128> {
    10u128.checked_pow(exponent)
}

/// Scales a non negative `value` to integer units at `precision` decimal
/// places, truncating toward zero, through its mantissa and scale so no
/// intermediate `Decimal` can overflow.
///
/// # Errors
///
/// Returns [`MMError::NumericalError`] if `value` is negative or the
/// result does not fit in a `u128`.
fn decimal_to_units(value: Decimal, precision: u32) -> MMResult<u128> {
    let out_of_range =
        || MMError::NumericalError(format!("{value} is out of range at precision {precision}"));
    let mantissa = u128::try_from(value.mantissa()).map_err(|_| out_of_range())?;
    let scale = value.scale();
    if precision >= scale {
        let factor = precision
            .checked_sub(scale)
            .and_then(pow10)
            .ok_or_else(out_of_range)?;
        mantissa.checked_mul(factor).ok_or_else(out_of_range)
    } else {
        // `scale - precision` is at most 28, so the power always fits.
        let divisor = scale
            .checked_sub(precision)
            .and_then(pow10)
            .ok_or_else(out_of_range)?;
        mantissa.checked_div(divisor).ok_or_else(out_of_range)
    }
}

/// Builds the `Decimal` worth `raw` integer units at `precision` decimal
/// places. Accepts any `u128` whose value is inside the `Decimal` range.
///
/// # Errors
///
/// Returns [`MMError::InvalidConfiguration`] if `precision` is above
/// [`MAX_PRECISION`], or [`MMError::NumericalError`] if the value is
/// outside the `Decimal` range.
fn units_to_decimal(raw: u128, precision: u32) -> MMResult<Decimal> {
    if precision > MAX_PRECISION {
        return Err(MMError::InvalidConfiguration(format!(
            "precision must be at most {MAX_PRECISION}, got {precision}"
        )));
    }
    let out_of_range = || {
        MMError::NumericalError(format!(
            "raw value {raw} at precision {precision} is out of the decimal range"
        ))
    };
    let unit = pow10(precision).ok_or_else(out_of_range)?;
    let (whole, fraction) = (
        raw.checked_div(unit).ok_or_else(out_of_range)?,
        raw.checked_rem(unit).ok_or_else(out_of_range)?,
    );
    let whole = Decimal::from_u128(whole).ok_or_else(out_of_range)?;
    // `fraction < 10^19` always fits the 96 bit mantissa.
    let fraction = i128::try_from(fraction)
        .ok()
        .and_then(|f| Decimal::try_from_i128_with_scale(f, precision).ok())
        .ok_or_else(out_of_range)?;
    let value = whole.checked_add(fraction).ok_or_else(out_of_range)?;
    // The 96 bit mantissa can round a large raw value with a fractional
    // part; reject anything that does not convert back to the same raw.
    if decimal_to_units(value, precision).ok() != Some(raw) {
        return Err(MMError::NumericalError(format!(
            "raw value {raw} at precision {precision} is not exactly representable"
        )));
    }
    Ok(value)
}

impl OrderBookConnectorConfig {
    /// Validates the configuration.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] if `price_precision` or
    /// `quantity_precision` is above [`MAX_PRECISION`].
    pub fn validate(&self) -> MMResult<()> {
        for (name, precision) in [
            ("price_precision", self.price_precision),
            ("quantity_precision", self.quantity_precision),
        ] {
            if precision > MAX_PRECISION {
                return Err(MMError::InvalidConfiguration(format!(
                    "{name} must be at most {MAX_PRECISION}, got {precision}"
                )));
            }
        }
        Ok(())
    }
}

/// Maps a poisoned lock to [`MMError::LockPoisoned`].
/// Times `cancel_all_orders` follows a re-mapped order before giving up.
const MAX_CANCEL_REMAPS: usize = 8;

fn poisoned<T>(_: std::sync::PoisonError<T>) -> MMError {
    MMError::LockPoisoned("orderbook connector".to_string())
}

impl Default for OrderBookConnectorConfig {
    fn default() -> Self {
        Self {
            price_precision: 2,
            quantity_precision: 8,
            fee_rate: Decimal::ZERO,
            fee_currency: "USD".to_string(),
        }
    }
}

/// Exchange connector implementation for OrderBook-rs.
///
/// This connector wraps an `orderbook-rs` `OrderBook` and implements
/// the `ExchangeConnector` trait for integration with market making strategies.
pub struct OrderBookConnector {
    /// The underlying order book.
    order_book: Arc<DefaultOrderBook>,
    /// Configuration.
    config: OrderBookConnectorConfig,
    /// Order ID counter for generating unique IDs.
    order_id_counter: AtomicU64,
    /// Mapping from our OrderId to orderbook-rs OrderId.
    order_mapping: std::sync::RwLock<HashMap<String, OBOrderId>>,
    /// Simulated balances for testing.
    balances: std::sync::RwLock<HashMap<String, Decimal>>,
}

impl OrderBookConnector {
    /// Creates a new OrderBook connector for the given symbol.
    ///
    /// # Arguments
    ///
    /// * `symbol` - The trading symbol (e.g., "BTC-USD")
    ///
    /// # Returns
    ///
    /// A new `OrderBookConnector` instance.
    #[must_use]
    pub fn new(symbol: &str) -> Self {
        Self {
            order_book: Arc::new(OrderBook::new(symbol)),
            config: OrderBookConnectorConfig::default(),
            order_id_counter: AtomicU64::new(1),
            order_mapping: std::sync::RwLock::new(HashMap::new()),
            balances: std::sync::RwLock::new(HashMap::new()),
        }
    }

    /// Creates a new OrderBook connector with custom configuration.
    ///
    /// # Arguments
    ///
    /// * `symbol` - The trading symbol
    /// * `config` - The connector configuration
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] if a precision is above
    /// [`MAX_PRECISION`].
    pub fn with_config(symbol: &str, config: OrderBookConnectorConfig) -> MMResult<Self> {
        Self::from_orderbook(Arc::new(OrderBook::new(symbol)), config)
    }

    /// Creates a new OrderBook connector wrapping an existing order book.
    ///
    /// # Arguments
    ///
    /// * `order_book` - The existing order book to wrap
    /// * `config` - The connector configuration
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] if a precision is above
    /// [`MAX_PRECISION`].
    pub fn from_orderbook(
        order_book: Arc<DefaultOrderBook>,
        config: OrderBookConnectorConfig,
    ) -> MMResult<Self> {
        config.validate()?;
        Ok(Self {
            order_book,
            config,
            order_id_counter: AtomicU64::new(1),
            order_mapping: std::sync::RwLock::new(HashMap::new()),
            balances: std::sync::RwLock::new(HashMap::new()),
        })
    }

    /// Returns a reference to the underlying order book.
    #[must_use]
    pub fn order_book(&self) -> &DefaultOrderBook {
        &self.order_book
    }

    /// Returns the symbol of this connector.
    #[must_use]
    pub fn symbol(&self) -> &str {
        self.order_book.symbol()
    }

    /// Sets a simulated balance for testing.
    ///
    /// # Arguments
    ///
    /// * `asset` - The asset symbol
    /// * `balance` - The balance amount
    /// # Errors
    ///
    /// Returns [`MMError::LockPoisoned`] if the lock is poisoned.
    pub fn set_balance(&self, asset: &str, balance: Decimal) -> MMResult<()> {
        let mut balances = self.balances.write().map_err(poisoned)?;
        balances.insert(asset.to_string(), balance);
        Ok(())
    }

    /// Generates a new unique order ID.
    fn next_order_id(&self) -> (OrderId, OBOrderId) {
        let id = self.order_id_counter.fetch_add(1, Ordering::SeqCst);
        let order_id = OrderId::new(id.to_string());
        let ob_order_id = OBOrderId::sequential(id);
        (order_id, ob_order_id)
    }

    /// Generates a new unique orderbook-rs order ID from the shared counter.
    fn next_ob_order_id(&self) -> OBOrderId {
        OBOrderId::sequential(self.order_id_counter.fetch_add(1, Ordering::SeqCst))
    }

    /// Converts a supported direction to orderbook-rs Side.
    fn convert_side(dir: Dir) -> OBSide {
        match dir {
            Dir::Buy => OBSide::Buy,
            Dir::Sell => OBSide::Sell,
        }
    }

    /// Converts orderbook-rs Side to our Action.
    #[allow(dead_code)]
    fn convert_side_back(side: OBSide) -> Action {
        match side {
            OBSide::Buy => Action::Buy,
            OBSide::Sell => Action::Sell,
        }
    }

    /// Converts our TimeInForce to orderbook-rs TimeInForce.
    fn convert_tif(tif: TimeInForce) -> OBTimeInForce {
        match tif {
            TimeInForce::GoodTilCancel => OBTimeInForce::Gtc,
            TimeInForce::ImmediateOrCancel => OBTimeInForce::Ioc,
            TimeInForce::FillOrKill => OBTimeInForce::Fok,
            TimeInForce::GoodTilTime(ts) => OBTimeInForce::Gtd(ts),
        }
    }

    /// Converts a Decimal price to the raw `u128` used by orderbook-rs.
    ///
    /// Rejects a price that is not positive, that rounds to zero ticks, or
    /// that does not fit: a zero raw price would make a sell marketable.
    fn price_to_u128(&self, price: Decimal) -> MMResult<u128> {
        let precision = self.config.price_precision;
        let not_positive = || {
            MMError::NumericalError(format!(
                "price {price} is not a positive raw price at precision {precision}"
            ))
        };
        if price <= Decimal::ZERO {
            return Err(not_positive());
        }
        match decimal_to_units(price, precision)? {
            0 => Err(not_positive()),
            raw => Ok(raw),
        }
    }

    /// Converts a raw `u128` price from orderbook-rs to Decimal.
    fn u128_to_price(&self, price: u128) -> MMResult<Decimal> {
        units_to_decimal(price, self.config.price_precision)
    }

    /// Converts a Decimal quantity to u64 for orderbook-rs.
    ///
    /// Rejects a quantity that is negative (checked before truncation, so
    /// a tiny negative quantity is rejected too) or does not fit in a `u64`.
    fn quantity_to_u64(&self, quantity: Decimal) -> MMResult<u64> {
        if quantity < Decimal::ZERO {
            return Err(MMError::NumericalError(format!(
                "quantity {quantity} is negative"
            )));
        }
        let raw = decimal_to_units(quantity, self.config.quantity_precision)?;
        if raw == 0 {
            // A zero or sub-lot quantity would cancel a resting order on modify
            // or submit an empty order, so it is rejected before the book changes.
            return Err(MMError::NumericalError(format!(
                "quantity {quantity} rounds to zero lots"
            )));
        }
        u64::try_from(raw).map_err(|_| {
            MMError::NumericalError(format!("quantity {quantity} does not fit in u64"))
        })
    }

    /// Converts a u64 quantity from orderbook-rs to Decimal.
    fn u64_to_quantity(&self, quantity: u64) -> MMResult<Decimal> {
        units_to_decimal(u128::from(quantity), self.config.quantity_precision)
    }

    /// Gets the current timestamp in milliseconds.
    fn current_timestamp() -> u64 {
        orderbook_rs::current_time_millis()
    }
}

impl OrderBookConnector {
    /// Cancels every tracked order in `order_ids` with `cancel`.
    ///
    /// An order leaves the mapping only when it is no longer resting: its
    /// cancel succeeded, it was already gone, or the book removed it and then
    /// reported a level fault. An order whose cancel failed while it still
    /// rests stays tracked, so it can be cancelled again. An entry is
    /// removed only while it still maps to the book order just cancelled: if
    /// a concurrent modify re-mapped it, the new book order is cancelled
    /// instead, and an entry that vanished meanwhile is left to the call
    /// that settled it. An order the book no longer knows (filled or
    /// cancelled outside the connector) is settled too: it is unmapped and
    /// left out of the cancelled responses, never reported as a failure.
    ///
    /// # Errors
    ///
    /// Returns the first cancel failure as
    /// [`MMError::InvalidPositionUpdate`] after every order was attempted,
    /// or [`MMError::LockPoisoned`] if the mapping lock is poisoned.
    fn cancel_tracked<F>(
        &self,
        order_ids: Vec<(String, OBOrderId)>,
        cancel: F,
    ) -> MMResult<Vec<OrderResponse>>
    where
        F: Fn(OBOrderId) -> Result<(), OrderBookError>,
    {
        let mut responses = Vec::new();
        let mut first_error = None;

        for (id_str, snapshot_id) in order_ids {
            let mut ob_id = snapshot_id;
            let mut remaps = 0usize;
            let outcome = loop {
                let outcome = cancel(ob_id);
                let removed = match &outcome {
                    Ok(())
                    | Err(OrderBookError::OrderRemovedWithLevelFault { .. })
                    | Err(OrderBookError::OrderNotFound(_)) => true,
                    Err(_) => false,
                };
                // Filled or cancelled outside the connector: settled.
                let not_found = matches!(outcome, Err(OrderBookError::OrderNotFound(_)));
                let mut mapping = self.order_mapping.write().map_err(poisoned)?;
                match mapping.get(id_str.as_str()).copied() {
                    // Still mapped to the order just cancelled: settle it.
                    Some(current) if current == ob_id => {
                        if removed {
                            mapping.remove(id_str.as_str());
                        }
                        if not_found {
                            break None;
                        }
                        break Some(outcome.map_err(|e| e.to_string()));
                    }
                    // Re-mapped meanwhile (a modify replaced the book order):
                    // the live order is the new one, so cancel that instead.
                    Some(current) => {
                        drop(mapping);
                        remaps = remaps.saturating_add(1);
                        if remaps > MAX_CANCEL_REMAPS {
                            break Some(Err(format!(
                                "order kept being replaced while cancelling ({remaps} times)"
                            )));
                        }
                        ob_id = current;
                    }
                    // No longer tracked: another call settled it.
                    None => break None,
                }
            };
            match outcome {
                None => {}
                Some(Ok(())) => responses.push(OrderResponse {
                    order_id: OrderId::new(id_str),
                    client_order_id: None,
                    status: OrderStatus::Cancelled {
                        filled_qty: Decimal::ZERO,
                    },
                    timestamp: Self::current_timestamp(),
                }),
                Some(Err(e)) if first_error.is_none() => {
                    first_error = Some(MMError::InvalidPositionUpdate(format!(
                        "cancel of order {id_str} failed: {e}"
                    )));
                }
                Some(Err(_)) => {}
            }
        }

        match first_error {
            Some(error) => Err(error),
            None => Ok(responses),
        }
    }
}

#[async_trait]
impl ExchangeConnector for OrderBookConnector {
    async fn submit_order(&self, request: OrderRequest) -> MMResult<OrderResponse> {
        // Validated before an order id is consumed or the book is touched.
        let dir = Dir::try_from(request.side)?;
        let (order_id, ob_order_id) = self.next_order_id();
        let ob_side = Self::convert_side(dir);
        let ob_tif = Self::convert_tif(request.time_in_force);
        let quantity = self.quantity_to_u64(request.quantity)?;

        let result = match request.order_type {
            OrderType::Market => {
                // Submit market order
                self.order_book
                    .submit_market_order(ob_order_id, quantity, ob_side)
                    .map_err(|e| MMError::InvalidPositionUpdate(e.to_string()))?;

                // Market orders are immediately filled or rejected
                OrderResponse {
                    order_id: order_id.clone(),
                    client_order_id: request.client_order_id,
                    status: OrderStatus::Filled {
                        filled_qty: request.quantity,
                        avg_price: request.price.unwrap_or(Decimal::ZERO),
                    },
                    timestamp: Self::current_timestamp(),
                }
            }
            OrderType::Limit => {
                let price = request.price.ok_or_else(|| {
                    MMError::InvalidConfiguration("Limit order requires price".to_string())
                })?;
                let ob_price = self.price_to_u128(price)?;

                // The mapping lock is held across the book update so no
                // other caller can see the order in the book but not in
                // the mapping, and a poisoned lock fails before the book
                // is touched.
                let mut mapping = self.order_mapping.write().map_err(poisoned)?;
                self.order_book
                    .add_limit_order(ob_order_id, ob_price, quantity, ob_side, ob_tif, None)
                    .map_err(|e| MMError::InvalidPositionUpdate(e.to_string()))?;
                mapping.insert(order_id.as_str().to_string(), ob_order_id);
                drop(mapping);

                OrderResponse {
                    order_id: order_id.clone(),
                    client_order_id: request.client_order_id,
                    status: OrderStatus::Open {
                        filled_qty: Decimal::ZERO,
                    },
                    timestamp: Self::current_timestamp(),
                }
            }
            OrderType::PostOnly => {
                let price = request.price.ok_or_else(|| {
                    MMError::InvalidConfiguration("PostOnly order requires price".to_string())
                })?;
                let ob_price = self.price_to_u128(price)?;

                // The mapping lock is held across the book update so no
                // other caller can see the order in the book but not in
                // the mapping, and a poisoned lock fails before the book
                // is touched.
                let mut mapping = self.order_mapping.write().map_err(poisoned)?;
                self.order_book
                    .add_post_only_order(ob_order_id, ob_price, quantity, ob_side, ob_tif, None)
                    .map_err(|e| MMError::InvalidPositionUpdate(e.to_string()))?;
                mapping.insert(order_id.as_str().to_string(), ob_order_id);
                drop(mapping);

                OrderResponse {
                    order_id: order_id.clone(),
                    client_order_id: request.client_order_id,
                    status: OrderStatus::Open {
                        filled_qty: Decimal::ZERO,
                    },
                    timestamp: Self::current_timestamp(),
                }
            }
        };

        Ok(result)
    }

    async fn cancel_order(&self, order_id: &OrderId) -> MMResult<OrderResponse> {
        let ob_order_id = {
            let mapping = self.order_mapping.read().map_err(poisoned)?;
            mapping.get(order_id.as_str()).copied().ok_or_else(|| {
                MMError::InvalidPositionUpdate(format!("Order not found: {}", order_id))
            })?
        };

        self.order_book
            .cancel_order(ob_order_id)
            .map_err(|e| MMError::InvalidPositionUpdate(e.to_string()))?;

        // Remove from mapping
        {
            let mut mapping = self.order_mapping.write().map_err(poisoned)?;
            mapping.remove(order_id.as_str());
        }

        Ok(OrderResponse {
            order_id: order_id.clone(),
            client_order_id: None,
            status: OrderStatus::Cancelled {
                filled_qty: Decimal::ZERO,
            },
            timestamp: Self::current_timestamp(),
        })
    }

    async fn modify_order(
        &self,
        order_id: &OrderId,
        new_price: Option<Decimal>,
        new_quantity: Option<Decimal>,
    ) -> MMResult<OrderResponse> {
        // The mapping write lock is held through the whole cancel/replace
        // so concurrent callers never see a stale or missing mapping.
        let mut mapping = self.order_mapping.write().map_err(poisoned)?;
        let ob_order_id = mapping.get(order_id.as_str()).copied().ok_or_else(|| {
            MMError::InvalidPositionUpdate(format!("Order not found: {}", order_id))
        })?;

        // For order modification, we cancel and re-add the order
        // This is a common pattern when the underlying system doesn't support atomic modifications

        // Get the current order to extract its details
        let order = self.order_book.get_order(ob_order_id).ok_or_else(|| {
            MMError::InvalidPositionUpdate(format!("Order not found in book: {}", order_id))
        })?;

        // Extract current order details
        let (current_price, current_quantity, side, tif) = match order.as_ref() {
            orderbook_rs::OrderType::Standard {
                price,
                quantity,
                side,
                time_in_force,
                ..
            } => (price.as_u128(), quantity.as_u64(), *side, *time_in_force),
            orderbook_rs::OrderType::PostOnly {
                price,
                quantity,
                side,
                time_in_force,
                ..
            } => (price.as_u128(), quantity.as_u64(), *side, *time_in_force),
            _ => {
                return Err(MMError::InvalidPositionUpdate(
                    "Unsupported order type for modification".to_string(),
                ));
            }
        };

        // Convert the new values before touching the book, so a bad value
        // leaves the existing order resting.
        let final_price = match new_price {
            Some(p) => self.price_to_u128(p)?,
            None => current_price,
        };
        let final_quantity = match new_quantity {
            Some(q) => self.quantity_to_u64(q)?,
            None => current_quantity,
        };

        // Cancel the existing order
        self.order_book
            .cancel_order(ob_order_id)
            .map_err(|e| MMError::InvalidPositionUpdate(e.to_string()))?;

        // Generate new order ID for the replacement order
        let new_ob_order_id = self.next_ob_order_id();

        self.order_book
            .add_limit_order(
                new_ob_order_id,
                final_price,
                final_quantity,
                side,
                tif,
                None,
            )
            .map_err(|e| MMError::InvalidPositionUpdate(e.to_string()))?;

        // Update mapping with new order ID
        mapping.insert(order_id.as_str().to_string(), new_ob_order_id);
        drop(mapping);

        Ok(OrderResponse {
            order_id: order_id.clone(),
            client_order_id: None,
            status: OrderStatus::Open {
                filled_qty: Decimal::ZERO,
            },
            timestamp: Self::current_timestamp(),
        })
    }

    async fn get_order_status(&self, order_id: &OrderId) -> MMResult<OrderResponse> {
        let ob_order_id = {
            let mapping = self.order_mapping.read().map_err(poisoned)?;
            mapping.get(order_id.as_str()).copied().ok_or_else(|| {
                MMError::InvalidPositionUpdate(format!("Order not found: {}", order_id))
            })?
        };

        let order = self.order_book.get_order(ob_order_id).ok_or_else(|| {
            MMError::InvalidPositionUpdate(format!("Order not found in book: {}", order_id))
        })?;

        // Order exists in the book, so it's open
        let _ = order; // Acknowledge we have the order

        Ok(OrderResponse {
            order_id: order_id.clone(),
            client_order_id: None,
            status: OrderStatus::Open {
                filled_qty: Decimal::ZERO,
            },
            timestamp: Self::current_timestamp(),
        })
    }

    async fn get_open_orders(&self, _symbol: &str) -> MMResult<Vec<OrderResponse>> {
        let mapping = self.order_mapping.read().map_err(poisoned)?;
        let mut orders = Vec::new();

        for id_str in mapping.keys() {
            let id_string = id_str.to_string();
            orders.push(OrderResponse {
                order_id: OrderId::new(id_string),
                client_order_id: None,
                status: OrderStatus::Open {
                    filled_qty: Decimal::ZERO,
                },
                timestamp: Self::current_timestamp(),
            });
        }

        Ok(orders)
    }

    async fn cancel_all_orders(&self, _symbol: &str) -> MMResult<Vec<OrderResponse>> {
        let order_ids: Vec<(String, OBOrderId)> = {
            let mapping = self.order_mapping.read().map_err(poisoned)?;
            mapping
                .iter()
                .map(|(k, v): (&String, &OBOrderId)| (k.clone(), *v))
                .collect()
        };

        self.cancel_tracked(order_ids, |ob_id| {
            self.order_book.cancel_order(ob_id).map(|_| ())
        })
    }

    async fn get_orderbook(&self, _symbol: &str, depth: usize) -> MMResult<OrderBookSnapshot> {
        let snapshot: OBSnapshot = self
            .order_book
            .create_snapshot(depth)
            .map_err(|e| MMError::InvalidMarketState(e.to_string()))?;

        let bids = snapshot
            .bids
            .iter()
            .map(|level| {
                Ok(BookLevel::new(
                    self.u128_to_price(level.price().as_u128())?,
                    self.u64_to_quantity(level.visible_quantity().as_u64())?,
                ))
            })
            .collect::<MMResult<Vec<BookLevel>>>()?;
        let asks = snapshot
            .asks
            .iter()
            .map(|level| {
                Ok(BookLevel::new(
                    self.u128_to_price(level.price().as_u128())?,
                    self.u64_to_quantity(level.visible_quantity().as_u64())?,
                ))
            })
            .collect::<MMResult<Vec<BookLevel>>>()?;

        Ok(OrderBookSnapshot {
            symbol: self.order_book.symbol().to_string(),
            bids,
            asks,
            timestamp: snapshot.timestamp,
        })
    }

    async fn get_balance(&self, asset: &str) -> MMResult<Decimal> {
        let balances = self.balances.read().map_err(poisoned)?;
        Ok(balances.get(asset).copied().unwrap_or(Decimal::ZERO))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dec;

    #[tokio::test]
    async fn test_connector_creation() {
        let connector = OrderBookConnector::new("BTC-USD");
        assert_eq!(connector.symbol(), "BTC-USD");
    }

    fn config(price_precision: u32, quantity_precision: u32) -> OrderBookConnectorConfig {
        OrderBookConnectorConfig {
            price_precision,
            quantity_precision,
            ..OrderBookConnectorConfig::default()
        }
    }

    #[test]
    fn test_with_config_precision_above_max_returns_err() {
        for cfg in [config(20, 8), config(2, 20)] {
            assert!(matches!(
                OrderBookConnector::with_config("BTC-USD", cfg),
                Err(MMError::InvalidConfiguration(_))
            ));
        }
        assert!(OrderBookConnector::with_config("BTC-USD", config(19, 19)).is_ok());
    }

    #[test]
    fn test_from_orderbook_precision_above_max_returns_err() {
        let book = Arc::new(OrderBook::new("BTC-USD"));
        assert!(matches!(
            OrderBookConnector::from_orderbook(book, config(u32::MAX, 8)),
            Err(MMError::InvalidConfiguration(_))
        ));
    }

    #[test]
    fn test_raw_to_decimal_beyond_range_returns_err() {
        let connector = OrderBookConnector::new("BTC-USD");
        assert!(matches!(
            connector.u128_to_price(u128::MAX),
            Err(MMError::NumericalError(_))
        ));
        assert_eq!(connector.u128_to_price(12345).unwrap(), dec!(123.45));
        assert_eq!(connector.u64_to_quantity(150_000_000).unwrap(), dec!(1.5));
    }

    #[test]
    fn test_price_to_u128_non_positive_or_huge_returns_err() {
        let connector = OrderBookConnector::new("BTC-USD");
        for price in [dec!(0), dec!(-1), dec!(0.001)] {
            assert!(matches!(
                connector.price_to_u128(price),
                Err(MMError::NumericalError(_))
            ));
        }
        assert_eq!(connector.price_to_u128(dec!(1.23)).unwrap(), 123);
        // Decimal::MAX * 10^19 does not fit in a u128.
        let fine = OrderBookConnector::with_config("BTC-USD", config(19, 8)).unwrap();
        assert!(matches!(
            fine.price_to_u128(Decimal::MAX),
            Err(MMError::NumericalError(_))
        ));
    }

    #[test]
    fn test_units_to_decimal_inexact_raw_returns_err_exact_large_raw_converts() {
        assert!(matches!(
            units_to_decimal(u128::MAX, 19),
            Err(MMError::NumericalError(_))
        ));
        let raw = 10u128.pow(29);
        let value = units_to_decimal(raw, 19).unwrap();
        assert_eq!(value, dec!(10000000000));
        assert_eq!(decimal_to_units(value, 19).unwrap(), raw);
    }

    #[test]
    fn test_price_to_u128_beyond_decimal_mantissa_round_trips() {
        // Decimal::MAX * 100 is above 2^96 but a valid u128 raw price.
        let connector = OrderBookConnector::new("BTC-USD");
        let raw = connector.price_to_u128(Decimal::MAX).unwrap();
        assert_eq!(raw, 7_922_816_251_426_433_759_354_395_033_500);
        assert_eq!(connector.u128_to_price(raw).unwrap(), Decimal::MAX);
        // A raw value above 2^96 whose price is inside the Decimal range.
        let big = 10u128.pow(30);
        assert_eq!(
            connector.u128_to_price(big).unwrap(),
            Decimal::from_u128(10u128.pow(28)).unwrap()
        );
    }

    #[test]
    fn test_quantity_to_u64_tiny_negative_returns_err() {
        let connector = OrderBookConnector::new("BTC-USD");
        assert!(matches!(
            connector.quantity_to_u64(dec!(-0.000000001)),
            Err(MMError::NumericalError(_))
        ));
        assert!(matches!(
            connector.quantity_to_u64(dec!(0.000000001)),
            Err(MMError::NumericalError(_))
        ));
    }

    #[tokio::test]
    async fn test_submit_order_poisoned_mapping_leaves_book_untouched() {
        let connector = OrderBookConnector::new("BTC-USD");
        std::thread::scope(|scope| {
            let _ = scope
                .spawn(|| {
                    let _guard = connector.order_mapping.write().unwrap();
                    panic!("poison the mapping");
                })
                .join();
        });
        let request = OrderRequest::limit_buy("BTC-USD", dec!(50000.0), dec!(1.0));
        assert!(matches!(
            connector.submit_order(request).await,
            Err(MMError::LockPoisoned(_))
        ));
        assert!(connector.order_book().best_bid().is_none());
    }

    #[test]
    fn test_quantity_to_u64_negative_or_huge_returns_err() {
        let connector = OrderBookConnector::new("BTC-USD");
        for quantity in [dec!(-1), dec!(1_000_000_000_000)] {
            assert!(matches!(
                connector.quantity_to_u64(quantity),
                Err(MMError::NumericalError(_))
            ));
        }
        assert!(matches!(
            connector.quantity_to_u64(dec!(0)),
            Err(MMError::NumericalError(_))
        ));
    }

    #[tokio::test]
    async fn test_submit_order_sell_at_zero_price_returns_err() {
        let connector = OrderBookConnector::new("BTC-USD");
        let request = OrderRequest::limit_sell("BTC-USD", dec!(0), dec!(1.0));
        assert!(matches!(
            connector.submit_order(request).await,
            Err(MMError::NumericalError(_))
        ));
    }

    #[tokio::test]
    async fn test_modify_order_invalid_price_keeps_order() {
        let connector = OrderBookConnector::new("BTC-USD");
        let request = OrderRequest::limit_buy("BTC-USD", dec!(50000.0), dec!(1.0));
        let response = connector.submit_order(request).await.unwrap();
        let result = connector
            .modify_order(&response.order_id, Some(dec!(-5)), None)
            .await;
        assert!(matches!(result, Err(MMError::NumericalError(_))));
        let book = connector.get_orderbook("BTC-USD", 10).await.unwrap();
        assert_eq!(book.bids.len(), 1);
    }

    #[tokio::test]
    async fn test_modify_order_sublot_quantity_keeps_order() {
        let connector = OrderBookConnector::new("BTC-USD");
        let request = OrderRequest::limit_buy("BTC-USD", dec!(100), dec!(1));
        let response = connector.submit_order(request).await.unwrap();
        let result = connector
            .modify_order(&response.order_id, None, Some(dec!(0.000000001)))
            .await;
        assert!(matches!(result, Err(MMError::NumericalError(_))));
        let book = connector.get_orderbook("BTC-USD", 10).await.unwrap();
        assert_eq!(book.bids.len(), 1);
        assert!(connector.get_order_status(&response.order_id).await.is_ok());
    }

    #[tokio::test]
    async fn test_submit_limit_order() {
        let connector = OrderBookConnector::new("BTC-USD");

        let request = OrderRequest::limit_buy("BTC-USD", dec!(50000.0), dec!(1.0));
        let response = connector.submit_order(request).await.unwrap();

        assert!(response.status.is_open());
    }

    #[tokio::test]
    async fn test_submit_and_cancel_order() {
        let connector = OrderBookConnector::new("BTC-USD");

        let request = OrderRequest::limit_buy("BTC-USD", dec!(50000.0), dec!(1.0));
        let response = connector.submit_order(request).await.unwrap();
        let order_id = response.order_id;

        let cancel_response = connector.cancel_order(&order_id).await.unwrap();
        assert!(cancel_response.status.is_terminal());
    }

    #[tokio::test]
    async fn test_get_orderbook() {
        let connector = OrderBookConnector::new("BTC-USD");

        // Add some orders
        let buy_request = OrderRequest::limit_buy("BTC-USD", dec!(49000.0), dec!(1.0));
        connector.submit_order(buy_request).await.unwrap();

        let sell_request = OrderRequest::limit_sell("BTC-USD", dec!(51000.0), dec!(1.0));
        connector.submit_order(sell_request).await.unwrap();

        let snapshot = connector.get_orderbook("BTC-USD", 10).await.unwrap();

        assert_eq!(snapshot.symbol, "BTC-USD");
        assert!(!snapshot.bids.is_empty());
        assert!(!snapshot.asks.is_empty());
    }

    #[tokio::test]
    async fn test_balance() {
        let connector = OrderBookConnector::new("BTC-USD");

        // Initially zero
        let balance = connector.get_balance("BTC").await.unwrap();
        assert_eq!(balance, Decimal::ZERO);

        // Set balance
        connector.set_balance("BTC", dec!(10.0)).unwrap();
        let balance = connector.get_balance("BTC").await.unwrap();
        assert_eq!(balance, dec!(10.0));
    }

    #[tokio::test]
    async fn test_cancel_all_orders_keeps_failed_cancels_tracked() {
        let connector = OrderBookConnector::new("BTC-USD");
        let mut ids = Vec::new();
        for i in 0..3 {
            let price = dec!(50000.0) - Decimal::from(i * 100);
            let request = OrderRequest::limit_buy("BTC-USD", price, dec!(1.0));
            ids.push(connector.submit_order(request).await.unwrap().order_id);
        }
        // The cancel of the middle order fails while it still rests.
        let stale = ids[1].clone();
        let stale_ob_id = connector.order_mapping.read().unwrap()[stale.as_str()];
        let order_ids: Vec<(String, OBOrderId)> = connector
            .order_mapping
            .read()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        let result = connector.cancel_tracked(order_ids, |ob_id| {
            if ob_id == stale_ob_id {
                Err(OrderBookError::InvalidPriceLevel(0))
            } else {
                connector.order_book.cancel_order(ob_id).map(|_| ())
            }
        });
        let error = result.unwrap_err();
        assert!(
            matches!(error, MMError::InvalidPositionUpdate(_)),
            "{error}"
        );
        assert!(error.to_string().contains(stale.as_str()), "{error}");

        // The two cancelled orders left the mapping; the failed one stays.
        let mapping = connector.order_mapping.read().unwrap();
        assert_eq!(mapping.len(), 1);
        assert!(mapping.contains_key(stale.as_str()));
    }

    #[tokio::test]
    async fn test_cancel_all_orders_follows_a_remapped_order() {
        let connector = OrderBookConnector::new("BTC-USD");
        let a = connector
            .submit_order(OrderRequest::limit_buy("BTC-USD", dec!(50000.0), dec!(1.0)))
            .await
            .unwrap()
            .order_id;
        // A second book order that a modify would map `a` to.
        let replacement = connector
            .submit_order(OrderRequest::limit_buy("BTC-USD", dec!(49000.0), dec!(1.0)))
            .await
            .unwrap()
            .order_id;
        let ob2 = connector
            .order_mapping
            .write()
            .unwrap()
            .remove(replacement.as_str())
            .unwrap();
        let ob1 = connector.order_mapping.read().unwrap()[a.as_str()];
        let order_ids = vec![(a.as_str().to_string(), ob1)];
        let calls = std::sync::Mutex::new(Vec::new());
        let result = connector.cancel_tracked(order_ids, |ob_id| {
            calls.lock().unwrap().push(ob_id);
            if ob_id == ob1 {
                // The modify lands between the snapshot and this cancel.
                connector.order_book.cancel_order(ob1).map(|_| ())?;
                connector
                    .order_mapping
                    .write()
                    .unwrap()
                    .insert(a.as_str().to_string(), ob2);
                Ok(())
            } else {
                connector.order_book.cancel_order(ob_id).map(|_| ())
            }
        });
        let responses = result.unwrap();
        assert_eq!(*calls.lock().unwrap(), vec![ob1, ob2]);
        assert_eq!(responses.len(), 1);
        assert!(connector.order_mapping.read().unwrap().is_empty());
        // Nothing is left resting untracked.
        let book = connector.order_book.create_snapshot(10).unwrap();
        assert!(book.bids.is_empty(), "{:?}", book.bids);
    }

    #[tokio::test]
    async fn test_cancel_all_orders_unmaps_an_order_the_book_lost() {
        let connector = OrderBookConnector::new("BTC-USD");
        let gone = connector
            .submit_order(OrderRequest::limit_buy("BTC-USD", dec!(50000.0), dec!(1.0)))
            .await
            .unwrap()
            .order_id;
        let live = connector
            .submit_order(OrderRequest::limit_buy("BTC-USD", dec!(49000.0), dec!(1.0)))
            .await
            .unwrap()
            .order_id;
        let gone_ob = connector.order_mapping.read().unwrap()[gone.as_str()];
        let order_ids: Vec<(String, OBOrderId)> = connector
            .order_mapping
            .read()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        let responses = connector
            .cancel_tracked(order_ids, |ob_id| {
                if ob_id == gone_ob {
                    Err(OrderBookError::OrderNotFound(ob_id.to_string()))
                } else {
                    connector.order_book.cancel_order(ob_id).map(|_| ())
                }
            })
            .unwrap();
        // Settled, not a failure: only the live order is reported cancelled.
        assert_eq!(responses.len(), 1);
        assert_eq!(responses[0].order_id, live);
        assert!(connector.order_mapping.read().unwrap().is_empty());
        // A later call does not keep failing on it.
        assert!(
            connector
                .cancel_all_orders("BTC-USD")
                .await
                .unwrap()
                .is_empty()
        );
    }

    #[tokio::test]
    async fn test_cancel_all_orders_poisoned_mapping() {
        let connector = OrderBookConnector::new("BTC-USD");
        let request = OrderRequest::limit_buy("BTC-USD", dec!(50000.0), dec!(1.0));
        connector.submit_order(request).await.unwrap();
        std::thread::scope(|scope| {
            let poisoner = scope.spawn(|| {
                let _guard = connector.order_mapping.write().unwrap();
                panic!("poison the order mapping");
            });
            assert!(poisoner.join().is_err());
        });

        let error = connector.cancel_all_orders("BTC-USD").await.unwrap_err();
        assert!(matches!(error, MMError::LockPoisoned(_)), "{error}");
        // The order is still resting in the book.
        assert_eq!(
            connector.order_book.create_snapshot(10).unwrap().bids.len(),
            1
        );
    }

    #[tokio::test]
    async fn test_cancel_all_orders() {
        let connector = OrderBookConnector::new("BTC-USD");

        // Add multiple orders
        for i in 0..5 {
            let price = dec!(50000.0) - Decimal::from(i * 100);
            let request = OrderRequest::limit_buy("BTC-USD", price, dec!(1.0));
            connector.submit_order(request).await.unwrap();
        }

        let open_orders = connector.get_open_orders("BTC-USD").await.unwrap();
        assert_eq!(open_orders.len(), 5);

        let cancelled = connector.cancel_all_orders("BTC-USD").await.unwrap();
        assert_eq!(cancelled.len(), 5);

        let open_orders = connector.get_open_orders("BTC-USD").await.unwrap();
        assert!(open_orders.is_empty());
    }

    #[tokio::test]
    async fn test_submit_order_unsupported_action_returns_err_without_effects() {
        let connector = OrderBookConnector::new("BTC-USD");
        let fresh = OrderBookConnector::new("BTC-USD");

        for order_type in [OrderType::Limit, OrderType::PostOnly, OrderType::Market] {
            let mut request = OrderRequest::limit_buy("BTC-USD", dec!(50000.0), dec!(1.0));
            request.order_type = order_type;
            request.side = Action::Other;
            assert_eq!(
                connector.submit_order(request).await,
                Err(MMError::UnsupportedAction {
                    action: Action::Other
                })
            );
        }
        assert!(
            connector
                .get_open_orders("BTC-USD")
                .await
                .unwrap()
                .is_empty()
        );
        let book = connector.get_orderbook("BTC-USD", 10).await.unwrap();
        assert!(book.bids.is_empty() && book.asks.is_empty());

        // No order id was consumed.
        let valid = OrderRequest::limit_buy("BTC-USD", dec!(50000.0), dec!(1.0));
        let first = connector.submit_order(valid.clone()).await.unwrap();
        let expected = fresh.submit_order(valid).await.unwrap();
        assert_eq!(first.order_id, expected.order_id);
    }
}
