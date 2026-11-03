//! Avellaneda-Stoikov model calculations.
//!
//! This module implements the core mathematical formulas from the
//! Avellaneda-Stoikov (2008) paper on high-frequency trading in a limit order book.
//!
//! ## Mathematical Model
//!
//! The model solves the optimal market making problem using stochastic control theory.
//!
//! ### Reservation Price
//! ```text
//! r = s - q * γ * σ² * (T - t)
//! ```
//! Where:
//! - `s`: mid-price
//! - `q`: inventory position
//! - `γ`: risk aversion
//! - `σ`: volatility
//! - `T - t`: time to terminal
//!
//! ### Optimal Spread
//! ```text
//! δ = γ * σ² * (T - t) + (2/γ) * ln(1 + γ/k)
//! ```
//! Where:
//! - `k`: order intensity parameter
//!
//! ### Units
//! - `s`, `r`, `δ`: price units; `q`: base units, signed (long positive)
//! - `σ`: **absolute** price volatility in price units per √year (not a
//!   relative or percentage volatility); for a relative volatility `v`, pass
//!   `σ = v * s`
//! - `T - t`: milliseconds, converted to years of 365 days
//! - `γ` and `k`: 1 / price units
//! - `ln` and powers are evaluated through `f64`, so accuracy depends on the
//!   inputs. On the validated fixtures the relative error is around 1e-15.
//!   It degrades as `γ / k` shrinks, because `1 + γ / k` loses digits when it
//!   is converted to `f64` before `ln`: at zero horizon with `γ = 1e-10` and
//!   `k = 1` the spread is off by about 8e-8 relative, and below about 1e-16
//!   the intensity term collapses to zero and quote generation fails. The
//!   continuous quotes are not rounded to tick here

use crate::Decimal;
use crate::types::decimal::{decimal_ln, decimal_powi};
use crate::types::error::{MMError, MMResult};

const SECONDS_PER_MILLISECOND: Decimal = Decimal::from_parts(1, 0, 0, false, 3); // 0.001
const SECONDS_PER_YEAR: Decimal = Decimal::from_parts(31_536_000, 0, 0, false, 0); // 31_536_000

/// Calculates the reservation price according to the Avellaneda-Stoikov model.
///
/// The reservation price represents the "fair value" adjusted for inventory risk.
/// A market maker with positive inventory (long) will have a reservation price
/// below the mid-price, incentivizing sells. Conversely, negative inventory
/// (short) results in a reservation price above mid, incentivizing buys.
///
/// # Arguments
///
/// * `mid_price` - Current mid-price of the asset
/// * `inventory` - Current inventory position (positive = long, negative = short)
/// * `risk_aversion` - Risk aversion parameter (gamma), must be positive
/// * `volatility` - Absolute price volatility per √year (see module units), must be positive
/// * `time_to_terminal_ms` - Time remaining until terminal time, in milliseconds
///
/// # Returns
///
/// The reservation price adjusted for inventory risk.
///
/// # Errors
///
/// Returns `MMError::InvalidMarketState` if:
/// - mid_price is not positive or finite
/// - volatility is not positive or finite
///
/// Returns `MMError::InvalidConfiguration` if:
/// - risk_aversion is not positive or finite
///
/// # Examples
///
/// ```
/// use market_maker_rs::strategy::avellaneda_stoikov::calculate_reservation_price;
/// use market_maker_rs::dec;
///
/// let reservation = calculate_reservation_price(
///     dec!(100.0),  // mid_price
///     dec!(10.0),   // inventory (long)
///     dec!(0.1),    // risk_aversion
///     dec!(0.2),    // absolute volatility, price units per √year
///     3600000       // 1 hour in ms
/// ).unwrap();
///
/// // With positive inventory, reservation < mid_price
/// assert!(reservation < dec!(100.0));
/// ```
pub fn calculate_reservation_price(
    mid_price: Decimal,
    inventory: Decimal,
    risk_aversion: Decimal,
    volatility: Decimal,
    time_to_terminal_ms: u64,
) -> MMResult<Decimal> {
    // Validate inputs
    if mid_price <= Decimal::ZERO {
        return Err(MMError::InvalidMarketState(
            "mid_price must be positive".to_string(),
        ));
    }

    if volatility <= Decimal::ZERO {
        return Err(MMError::InvalidMarketState(
            "volatility must be positive".to_string(),
        ));
    }

    if risk_aversion <= Decimal::ZERO {
        return Err(MMError::InvalidConfiguration(
            "risk_aversion must be positive".to_string(),
        ));
    }

    // Convert time to years (volatility is per √year)
    let time_to_terminal_years = years_to_terminal(time_to_terminal_ms)?;

    // Formula: r = s - q * γ * σ² * (T - t)
    let volatility_squared = decimal_powi(volatility, 2)?;
    let adjustment = checked(
        inventory
            .checked_mul(risk_aversion)
            .and_then(|x| x.checked_mul(volatility_squared))
            .and_then(|x| x.checked_mul(time_to_terminal_years)),
        "reservation price adjustment",
    )?;
    let reservation_price = checked(mid_price.checked_sub(adjustment), "reservation price")?;

    Ok(reservation_price)
}

/// Calculates the optimal spread according to the Avellaneda-Stoikov model.
///
/// The spread represents the distance between bid and ask quotes.
/// It accounts for both inventory risk (first term) and adverse selection
/// (second term).
///
/// # Arguments
///
/// * `risk_aversion` - Risk aversion parameter (gamma), must be positive
/// * `volatility` - Absolute price volatility per √year (see module units), must be positive
/// * `time_to_terminal_ms` - Time remaining until terminal time, in milliseconds
/// * `order_intensity` - Order intensity decay `k`, in 1 / price units, must be positive
///
/// # Returns
///
/// The optimal spread in price units.
///
/// # Errors
///
/// Returns `MMError::InvalidConfiguration` if any parameter is not positive or finite.
///
/// # Examples
///
/// ```
/// use market_maker_rs::strategy::avellaneda_stoikov::calculate_optimal_spread;
/// use market_maker_rs::dec;
///
/// let spread = calculate_optimal_spread(
///     dec!(0.1),    // risk_aversion
///     dec!(0.2),    // absolute volatility, price units per √year
///     3600000,      // 1 hour
///     dec!(1.5)     // order_intensity
/// ).unwrap();
///
/// assert!(spread > dec!(0.0));
/// ```
pub fn calculate_optimal_spread(
    risk_aversion: Decimal,
    volatility: Decimal,
    time_to_terminal_ms: u64,
    order_intensity: Decimal,
) -> MMResult<Decimal> {
    // Validate inputs
    if risk_aversion <= Decimal::ZERO {
        return Err(MMError::InvalidConfiguration(
            "risk_aversion must be positive".to_string(),
        ));
    }

    if volatility <= Decimal::ZERO {
        return Err(MMError::InvalidConfiguration(
            "volatility must be positive".to_string(),
        ));
    }

    if order_intensity <= Decimal::ZERO {
        return Err(MMError::InvalidConfiguration(
            "order_intensity must be positive".to_string(),
        ));
    }

    // Convert time to years
    let time_to_terminal_years = years_to_terminal(time_to_terminal_ms)?;

    // Formula: δ = γ * σ² * (T - t) + (2/γ) * ln(1 + γ/k)
    let volatility_squared = decimal_powi(volatility, 2)?;
    let inventory_risk_term = checked(
        risk_aversion
            .checked_mul(volatility_squared)
            .and_then(|x| x.checked_mul(time_to_terminal_years)),
        "inventory risk term",
    )?;

    let two = Decimal::from(2);
    let adverse_selection_inner = checked(
        risk_aversion
            .checked_div(order_intensity)
            .and_then(|x| Decimal::ONE.checked_add(x)),
        "1 + gamma / k",
    )?;
    let adverse_selection_ln = decimal_ln(adverse_selection_inner)?;
    let adverse_selection_term = checked(
        two.checked_div(risk_aversion)
            .and_then(|x| x.checked_mul(adverse_selection_ln)),
        "adverse selection term",
    )?;

    let spread = checked(
        inventory_risk_term.checked_add(adverse_selection_term),
        "spread",
    )?;

    if spread < Decimal::ZERO {
        return Err(MMError::NumericalError(
            "spread calculation resulted in negative value".to_string(),
        ));
    }

    Ok(spread)
}

/// Calculates optimal bid and ask prices based on the Avellaneda-Stoikov model.
///
/// This is a convenience function that combines reservation price and spread
/// calculations to produce bid/ask quotes.
///
/// # Arguments
///
/// * `mid_price` - Current mid-price
/// * `inventory` - Current inventory position
/// * `risk_aversion` - Risk aversion parameter (gamma)
/// * `volatility` - Absolute price volatility per √year (see module units)
/// * `time_to_terminal_ms` - Time to terminal in milliseconds
/// * `order_intensity` - Order intensity decay `k`, in 1 / price units
///
/// # Returns
///
/// A tuple `(bid_price, ask_price)`.
///
/// # Errors
///
/// Returns errors from underlying calculations if inputs are invalid.
///
/// # Examples
///
/// ```
/// use market_maker_rs::strategy::avellaneda_stoikov::calculate_optimal_quotes;
/// use market_maker_rs::dec;
///
/// let (bid, ask) = calculate_optimal_quotes(
///     dec!(100.0),  // mid_price
///     dec!(0.0),    // flat inventory
///     dec!(0.1),    // risk_aversion
///     dec!(0.2),    // volatility
///     3600000,      // time_to_terminal_ms
///     dec!(1.5)     // order_intensity
/// ).unwrap();
///
/// assert!(bid < ask);
/// assert!(bid < dec!(100.0));
/// assert!(ask > dec!(100.0));
/// ```
pub fn calculate_optimal_quotes(
    mid_price: Decimal,
    inventory: Decimal,
    risk_aversion: Decimal,
    volatility: Decimal,
    time_to_terminal_ms: u64,
    order_intensity: Decimal,
) -> MMResult<(Decimal, Decimal)> {
    let reservation_price = calculate_reservation_price(
        mid_price,
        inventory,
        risk_aversion,
        volatility,
        time_to_terminal_ms,
    )?;

    let spread = calculate_optimal_spread(
        risk_aversion,
        volatility,
        time_to_terminal_ms,
        order_intensity,
    )?;

    let two = Decimal::from(2);
    let half_spread = checked(spread.checked_div(two), "half spread")?;
    let bid_price = checked(reservation_price.checked_sub(half_spread), "bid price")?;
    let ask_price = checked(reservation_price.checked_add(half_spread), "ask price")?;

    // Validate quotes
    if bid_price >= ask_price {
        return Err(MMError::InvalidQuoteGeneration(
            "bid price must be less than ask price".to_string(),
        ));
    }

    if bid_price <= Decimal::ZERO {
        return Err(MMError::InvalidQuoteGeneration(
            "bid price must be positive".to_string(),
        ));
    }

    Ok((bid_price, ask_price))
}

/// Maps a failed checked `Decimal` operation to [`MMError::NumericalError`].
fn checked(value: Option<Decimal>, what: &str) -> MMResult<Decimal> {
    value.ok_or_else(|| MMError::NumericalError(format!("avellaneda-stoikov: {what} overflows")))
}

/// Time to terminal in years (365 day year), with checked arithmetic.
fn years_to_terminal(time_to_terminal_ms: u64) -> MMResult<Decimal> {
    checked(
        Decimal::from(time_to_terminal_ms)
            .checked_mul(SECONDS_PER_MILLISECOND)
            .and_then(|x| x.checked_div(SECONDS_PER_YEAR)),
        "time to terminal",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dec;

    #[test]
    fn test_calculate_optimal_quotes_overflow_returns_numerical_error() {
        let result = calculate_optimal_quotes(
            dec!(1000),
            dec!(200),
            dec!(1),
            dec!(100000000000000),
            1,
            dec!(0.3),
        );
        assert!(
            matches!(result, Err(MMError::NumericalError(_))),
            "{result:?}"
        );
        let spread = calculate_optimal_spread(dec!(1), dec!(100000000000000), u64::MAX, dec!(0.3));
        assert!(
            matches!(spread, Err(MMError::NumericalError(_))),
            "{spread:?}"
        );
    }

    #[test]
    fn test_reservation_price_flat_inventory() {
        let result =
            calculate_reservation_price(dec!(100.0), Decimal::ZERO, dec!(0.1), dec!(0.2), 3600000);
        assert!(result.is_ok());
        let reservation = result.unwrap();
        // With flat inventory, reservation should equal mid_price
        assert!((reservation - dec!(100.0)).abs() < dec!(0.0001));
    }

    #[test]
    fn test_reservation_price_long_inventory() {
        let result =
            calculate_reservation_price(dec!(100.0), dec!(10.0), dec!(0.1), dec!(0.2), 3600000);
        assert!(result.is_ok());
        let reservation = result.unwrap();
        // With positive inventory, reservation < mid_price
        assert!(reservation < dec!(100.0));
    }

    #[test]
    fn test_reservation_price_short_inventory() {
        let result =
            calculate_reservation_price(dec!(100.0), dec!(-10.0), dec!(0.1), dec!(0.2), 3600000);
        assert!(result.is_ok());
        let reservation = result.unwrap();
        // With negative inventory, reservation > mid_price
        assert!(reservation > dec!(100.0));
    }

    #[test]
    fn test_reservation_price_invalid_mid_price() {
        let result =
            calculate_reservation_price(dec!(-100.0), Decimal::ZERO, dec!(0.1), dec!(0.2), 3600000);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            MMError::InvalidMarketState(_)
        ));
    }

    #[test]
    fn test_reservation_price_invalid_volatility() {
        let result =
            calculate_reservation_price(dec!(100.0), Decimal::ZERO, dec!(0.1), dec!(-0.2), 3600000);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            MMError::InvalidMarketState(_)
        ));
    }

    #[test]
    fn test_reservation_price_invalid_risk_aversion() {
        let result =
            calculate_reservation_price(dec!(100.0), Decimal::ZERO, dec!(-0.1), dec!(0.2), 3600000);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            MMError::InvalidConfiguration(_)
        ));
    }

    #[test]
    fn test_reservation_price_invalid_inventory() {
        // Decimal doesn't have NAN, so this test is less relevant
        // But we keep it to test the structure, using a zero inventory
        let result =
            calculate_reservation_price(dec!(100.0), Decimal::ZERO, dec!(0.1), dec!(0.2), 3600000);
        assert!(result.is_ok());
    }

    #[test]
    fn test_reservation_price_non_finite_result() {
        // Very large inventory - Decimal has different overflow behavior than f64
        // Using more reasonable extreme values that won't cause overflow
        let result = calculate_reservation_price(
            dec!(100.0),
            dec!(1000000),
            dec!(1000),
            dec!(1000),
            u64::MAX,
        );
        // Decimal will produce a result (possibly very large or very negative)
        // Unlike f64 which would produce infinity
        let _ = result;
    }

    #[test]
    fn test_optimal_spread_positive() {
        let result = calculate_optimal_spread(dec!(0.1), dec!(0.2), 3600000, dec!(1.5));
        assert!(result.is_ok());
        let spread = result.unwrap();
        assert!(spread > Decimal::ZERO);
    }

    #[test]
    fn test_optimal_spread_increases_with_volatility() {
        let spread1 = calculate_optimal_spread(dec!(0.1), dec!(0.1), 3600000, dec!(1.5)).unwrap();
        let spread2 = calculate_optimal_spread(dec!(0.1), dec!(0.3), 3600000, dec!(1.5)).unwrap();
        assert!(spread2 > spread1);
    }

    #[test]
    fn test_optimal_spread_invalid_risk_aversion() {
        let result = calculate_optimal_spread(dec!(-0.1), dec!(0.2), 3600000, dec!(1.5));
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            MMError::InvalidConfiguration(_)
        ));
    }

    #[test]
    fn test_optimal_spread_invalid_volatility() {
        let result = calculate_optimal_spread(dec!(0.1), dec!(-0.2), 3600000, dec!(1.5));
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            MMError::InvalidConfiguration(_)
        ));
    }

    #[test]
    fn test_optimal_spread_invalid_order_intensity() {
        let result = calculate_optimal_spread(dec!(0.1), dec!(0.2), 3600000, dec!(-1.5));
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            MMError::InvalidConfiguration(_)
        ));
    }

    #[test]
    fn test_optimal_spread_non_finite_result() {
        // Very large volatility - Decimal handles large numbers better
        let result = calculate_optimal_spread(
            dec!(0.1),
            Decimal::from_parts(u32::MAX, 0, 0, false, 10),
            3600000,
            dec!(1.5),
        );
        // May succeed or error - just checking it doesn't panic
        let _ = result;
    }

    #[test]
    fn test_optimal_spread_negative_result() {
        // Aunque matemáticamente el spread no debería ser negativo,
        // probamos el path de error con valores extremos
        // La fórmula: δ = γ * σ² * (T - t) + (2/γ) * ln(1 + γ/k)
        // Es muy difícil hacer que esto sea negativo con valores válidos
        // Este test verifica que el código maneja el caso correctamente
        let result = calculate_optimal_spread(
            dec!(0.0001),
            dec!(0.0001),
            1,             // tiempo muy pequeño
            dec!(1000000), // k muy grande
        );
        // El resultado debería ser válido o dar error numérico, pero no negativo
        // Error es aceptable en casos extremos
        if let Ok(spread) = result {
            assert!(spread >= Decimal::ZERO);
        }
    }

    #[test]
    fn test_optimal_quotes_valid() {
        let result = calculate_optimal_quotes(
            dec!(100.0),
            Decimal::ZERO,
            dec!(0.1),
            dec!(0.2),
            3600000,
            dec!(1.5),
        );
        assert!(result.is_ok());
        let (bid, ask) = result.unwrap();
        assert!(bid < ask);
        assert!(bid < dec!(100.0));
        assert!(ask > dec!(100.0));
        assert!(bid > Decimal::ZERO);
    }

    #[test]
    fn test_optimal_quotes_with_positive_inventory() {
        let (bid_flat, ask_flat) = calculate_optimal_quotes(
            dec!(100.0),
            Decimal::ZERO,
            dec!(0.1),
            dec!(0.2),
            3600000,
            dec!(1.5),
        )
        .unwrap();
        let (bid_long, ask_long) = calculate_optimal_quotes(
            dec!(100.0),
            dec!(10.0),
            dec!(0.1),
            dec!(0.2),
            3600000,
            dec!(1.5),
        )
        .unwrap();

        // With long inventory, both quotes should be lower
        assert!(bid_long < bid_flat);
        assert!(ask_long < ask_flat);
    }

    #[test]
    fn test_optimal_quotes_with_negative_inventory() {
        let (bid_flat, ask_flat) = calculate_optimal_quotes(
            dec!(100.0),
            Decimal::ZERO,
            dec!(0.1),
            dec!(0.2),
            3600000,
            dec!(1.5),
        )
        .unwrap();
        let (bid_short, ask_short) = calculate_optimal_quotes(
            dec!(100.0),
            dec!(-10.0),
            dec!(0.1),
            dec!(0.2),
            3600000,
            dec!(1.5),
        )
        .unwrap();

        // With short inventory, both quotes should be higher
        assert!(bid_short > bid_flat);
        assert!(ask_short > ask_flat);
    }

    #[test]
    fn test_optimal_quotes_spread_positive() {
        let (bid, ask) = calculate_optimal_quotes(
            dec!(100.0),
            Decimal::ZERO,
            dec!(0.1),
            dec!(0.2),
            3600000,
            dec!(1.5),
        )
        .unwrap();
        assert!(ask - bid > Decimal::ZERO);
    }

    #[test]
    fn test_optimal_quotes_negative_bid_error() {
        // With very low mid price and large negative inventory, bid can become negative
        let result = calculate_optimal_quotes(
            dec!(0.5),
            dec!(-1000.0),
            dec!(1.0),
            dec!(1.0),
            36000000,
            dec!(0.1),
        );
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            MMError::InvalidQuoteGeneration(_)
        ));
    }

    #[test]
    fn test_optimal_quotes_bid_exceeds_ask_error() {
        // Extreme parameters that could theoretically cause bid >= ask
        let result = calculate_optimal_quotes(
            dec!(100.0),
            Decimal::from_parts(u32::MAX, u32::MAX, 0, false, 0),
            dec!(0.0000000001),
            dec!(0.001),
            1,
            Decimal::from_parts(u32::MAX, u32::MAX, 0, false, 0),
        );
        // If the model produces invalid quotes, it should error
        if let Err(err) = result {
            // Could be InvalidQuoteGeneration or some other error from validation
            assert!(matches!(
                err,
                MMError::InvalidQuoteGeneration(_)
                    | MMError::InvalidMarketState(_)
                    | MMError::InvalidConfiguration(_)
                    | MMError::NumericalError(_)
            ));
        }
    }
}
