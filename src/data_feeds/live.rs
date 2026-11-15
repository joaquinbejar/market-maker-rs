//! Live exchange trades through `fin-stream` (feature `live-feeds`).
//!
//! `fin-stream` parses raw Binance, Coinbase, Alpaca and Polygon trade
//! messages into one exact-decimal [`NormalizedTick`]. This module maps that
//! tick into this crate's [`Trade`] and [`PriceUpdate`], `Decimal` to
//! `Decimal` with no float conversion, and maps a [`ReconnectConfig`] onto the
//! [`ReconnectPolicy`] of `fin-stream`'s WebSocket manager, so a live feed can
//! be built on the same types as [`MockDataFeed`](crate::data_feeds::MockDataFeed).
//!
//! # Example
//!
//! ```rust,ignore
//! use market_maker_rs::data_feeds::live::{Exchange, RawTick, normalize_trade};
//!
//! // One Binance `trade` stream message, as received on the socket.
//! let payload = serde_json::from_str(text)?;
//! let trade = normalize_trade(RawTick::new(Exchange::Binance, "BTCUSDT", payload))?;
//! ```

use std::time::Duration;

pub use fin_stream::tick::TradeSide as TickSide;
pub use fin_stream::{
    ConnectionConfig, Exchange, NormalizedTick, RawTick, ReconnectPolicy, TickNormalizer, WsManager,
};

use crate::data_feeds::feed::ReconnectConfig;
use crate::data_feeds::types::{DataSource, PriceUpdate, Trade, TradeSide};
use crate::types::error::{MMError, MMResult};

/// Normalizes one raw exchange trade message into a [`Trade`].
///
/// # Errors
///
/// Returns [`MMError::InvalidMarketState`] if the payload is missing a
/// required field, a price or quantity does not parse as a decimal, the price
/// is not positive or the quantity is negative.
pub fn normalize_trade(raw: RawTick) -> MMResult<Trade> {
    TickNormalizer::new()
        .normalize(raw)
        .map(Trade::from)
        .map_err(|e| MMError::InvalidMarketState(e.to_string()))
}

/// The tick's exchange timestamp in milliseconds, or the local receive time
/// when the exchange sent none.
fn tick_timestamp(tick: &NormalizedTick) -> u64 {
    tick.exchange_ts_ms.unwrap_or(tick.received_at_ms)
}

impl From<NormalizedTick> for Trade {
    /// Maps a normalized tick into a trade. A tick without an aggressor side
    /// (Alpaca, Polygon) becomes [`TradeSide::Unknown`], and one without a
    /// trade identifier gets an empty `trade_id`.
    fn from(tick: NormalizedTick) -> Self {
        let timestamp = tick_timestamp(&tick);
        let side = match tick.side {
            Some(TickSide::Buy) => TradeSide::Buy,
            Some(TickSide::Sell) => TradeSide::Sell,
            None => TradeSide::Unknown,
        };
        Trade::new(
            tick.symbol,
            tick.price,
            tick.quantity,
            side,
            timestamp,
            tick.trade_id.unwrap_or_default(),
        )
    }
}

impl From<NormalizedTick> for PriceUpdate {
    /// Maps a normalized tick into a last trade price update from
    /// [`DataSource::Exchange`], named after the tick's exchange.
    fn from(tick: NormalizedTick) -> Self {
        let timestamp = tick_timestamp(&tick);
        PriceUpdate::new(
            tick.symbol,
            tick.price,
            timestamp,
            DataSource::Exchange(tick.exchange.to_string()),
        )
    }
}

impl ReconnectConfig {
    /// The equivalent `fin-stream` reconnect policy: the same initial delay,
    /// cap and multiplier, and so the same delay for every attempt as
    /// [`ReconnectConfig::delay_for_attempt`]. No attempt limit maps to
    /// `u32::MAX` attempts.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] if `max_attempts` is
    /// `Some(0)` or `backoff_multiplier` is below 1, which `fin-stream`
    /// rejects.
    pub fn reconnect_policy(&self) -> MMResult<ReconnectPolicy> {
        ReconnectPolicy::new(
            self.max_attempts.unwrap_or(u32::MAX),
            Duration::from_millis(self.initial_delay_ms),
            Duration::from_millis(self.max_delay_ms),
            self.backoff_multiplier,
        )
        .map_err(|e| MMError::InvalidConfiguration(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    fn raw(exchange: Exchange, symbol: &str, payload: &str) -> RawTick {
        let mut raw = RawTick::new(exchange, symbol, serde_json::from_str(payload).unwrap());
        raw.received_at_ms = 42;
        raw
    }

    #[test]
    fn test_binance_trade() {
        let trade = normalize_trade(raw(
            Exchange::Binance,
            "BTCUSDT",
            r#"{"e":"trade","s":"BTCUSDT","t":12345,"p":"65000.10","q":"0.00100000","T":1700000000000,"m":true}"#,
        ))
        .unwrap();
        assert_eq!(trade.symbol, "BTCUSDT");
        assert_eq!(trade.price, dec!(65000.10));
        assert_eq!(trade.quantity, dec!(0.00100000));
        // The buyer was the maker, so the seller aggressed.
        assert_eq!(trade.side, TradeSide::Sell);
        assert_eq!(trade.timestamp, 1_700_000_000_000);
        assert_eq!(trade.trade_id, "12345");
    }

    #[test]
    fn test_coinbase_trade() {
        let trade = normalize_trade(raw(
            Exchange::Coinbase,
            "BTC-USD",
            r#"{"trade_id":"987","product_id":"BTC-USD","price":"64999.99","size":"0.5","side":"buy","time":"2023-11-14T22:13:20Z"}"#,
        ))
        .unwrap();
        assert_eq!(trade.price, dec!(64999.99));
        assert_eq!(trade.quantity, dec!(0.5));
        assert!(trade.is_buy());
        assert_eq!(trade.timestamp, 1_700_000_000_000);
        assert_eq!(trade.trade_id, "987");
    }

    #[test]
    fn test_side_and_timestamp_fallbacks() {
        // Alpaca sends no side; without `t` the receive time is used.
        let trade =
            normalize_trade(raw(Exchange::Alpaca, "AAPL", r#"{"p":"189.25","s":"100"}"#)).unwrap();
        assert_eq!(trade.side, TradeSide::Unknown);
        assert_eq!(trade.timestamp, 42);
        assert_eq!(trade.trade_id, "");
    }

    #[test]
    fn test_invalid_trade_is_rejected() {
        let missing = normalize_trade(raw(Exchange::Binance, "BTCUSDT", r#"{"q":"1"}"#));
        assert!(matches!(missing, Err(MMError::InvalidMarketState(_))));
        let zero = normalize_trade(raw(Exchange::Binance, "BTCUSDT", r#"{"p":"0","q":"1"}"#));
        assert!(matches!(zero, Err(MMError::InvalidMarketState(_))));
    }

    #[test]
    fn test_price_update_from_tick() {
        let tick = TickNormalizer::new()
            .normalize(raw(
                Exchange::Polygon,
                "AAPL",
                r#"{"p":"189.25","s":"100","t":1700000000000000000}"#,
            ))
            .unwrap();
        let update = PriceUpdate::from(tick);
        assert_eq!(update.symbol, "AAPL");
        assert_eq!(update.price, dec!(189.25));
        assert_eq!(update.timestamp, 1_700_000_000_000);
        assert_eq!(update.source, DataSource::Exchange("Polygon".to_string()));
    }

    #[test]
    fn test_reconnect_policy_matches_config() {
        let config = ReconnectConfig::default();
        let policy = config.reconnect_policy().unwrap();
        assert_eq!(policy.max_attempts(), u32::MAX);
        for attempt in 0..12 {
            assert_eq!(
                policy.backoff_for_attempt(attempt).as_millis() as u64,
                config.delay_for_attempt(attempt)
            );
        }

        let limited = ReconnectConfig::new(500, 10_000, 1.5, Some(3));
        let policy = limited.reconnect_policy().unwrap();
        for attempt in 0..6 {
            assert_eq!(policy.is_exhausted(attempt), !limited.should_retry(attempt));
        }
    }

    #[test]
    fn test_reconnect_policy_rejects_what_fin_stream_rejects() {
        let no_attempts = ReconnectConfig::new(1000, 30000, 2.0, Some(0));
        assert!(matches!(
            no_attempts.reconnect_policy(),
            Err(MMError::InvalidConfiguration(_))
        ));
        let shrinking = ReconnectConfig::new(1000, 30000, 0.5, None);
        assert!(matches!(
            shrinking.reconnect_policy(),
            Err(MMError::InvalidConfiguration(_))
        ));
    }
}
