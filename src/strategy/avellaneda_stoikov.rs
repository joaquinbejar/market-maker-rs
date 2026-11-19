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
//! - The continuous quotes are not rounded to tick here.
//!
//! ### Deterministic arithmetic
//! Everything except the logarithm is `Decimal` arithmetic: `σ²` is
//! `σ * σ`, and the products, sums and divisions are checked `Decimal`
//! operations. Products and sums of these inputs are exact; a `Decimal`
//! division rounds at 28 significant digits, so it is deterministic but not
//! exact. The only float step is `ln(1 + γ/k)`, evaluated as `x * L` with
//! `x = γ / k` (a `Decimal`) and
//!
//! ```text
//! L = log1p(x) / x
//! ```
//!
//! `x` goes to `f64` once, by parsing its decimal string (correctly rounded),
//! `log1p` is the pure software [`libm`] implementation, and the division is
//! a single IEEE 754 basic operation. No fused multiply add is used (never
//! `mul_add`), and Rust never contracts float operations implicitly. That is
//! what makes the value the same on every IEEE 754 target: determinism rests
//! on `libm` plus IEEE basic operations. `L` is then rounded half to even to
//! 12 significant digits
//! ([`LOG_RATIO_SIGNIFICANT_DIGITS`](crate::strategy::avellaneda_stoikov::LOG_RATIO_SIGNIFICANT_DIGITS))
//! and converted back to `Decimal` (for `γ / k` above about 4e17, where `L` is
//! below 1e-16, it keeps the 28 decimal places of `Decimal` instead); the rounding only lowers the odds that a
//! change of the last float bit (a different `libm` release, say) shows in
//! the quotes. The adverse selection term is then
//! `(2/γ) * x * L = (2/k) * L` in `Decimal`.
//!
//! `log1p` keeps full relative accuracy for a small `γ / k`, where
//! `ln(1 + γ/k)` would first lose the digits of `γ/k` in the sum, and the
//! significant digit rounding of `L` is relative, so the term carries a
//! relative error of about 1e-12 for `γ / k` up to about 4e17. The outputs are
//! bit identical on x86_64 and aarch64; `doc/fixtures/avellaneda_stoikov.json`
//! ([`AVELLANEDA_STOIKOV_FIXTURES`](crate::strategy::avellaneda_stoikov::AVELLANEDA_STOIKOV_FIXTURES))
//! pins them and is checked on both.

use crate::Decimal;
use crate::types::det_float::{
    decimal_to_f64, round_half_even_decimals, round_half_even_significant,
};
use crate::types::error::{MMError, MMResult};

const SECONDS_PER_MILLISECOND: Decimal = Decimal::from_parts(1, 0, 0, false, 3); // 0.001
const SECONDS_PER_YEAR: Decimal = Decimal::from_parts(31_536_000, 0, 0, false, 0); // 31_536_000

/// Significant digits kept, half to even, of the float ratio `log1p(γ/k) / (γ/k)`
/// (see the module docs).
pub const LOG_RATIO_SIGNIFICANT_DIGITS: u32 = 12;

/// Below this ratio (`γ / k` above about 4e17) the ratio keeps the 28 decimal places of
/// `Decimal` instead of [`LOG_RATIO_SIGNIFICANT_DIGITS`] significant digits.
const SIGNIFICANT_ROUNDING_FLOOR: f64 = 1e-16;

/// The fixtures of `doc/fixtures/avellaneda_stoikov.json`: a grid of inputs with the
/// expected reservation price, spread, bid and ask as decimal strings, identical on every
/// target. `doc/fixtures/AVELLANEDA_STOIKOV.md` describes the format.
pub const AVELLANEDA_STOIKOV_FIXTURES: &str =
    include_str!("../../doc/fixtures/avellaneda_stoikov.json");

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
/// Returns `MMError::NumericalError` if `σ²` or `q * γ * σ² * (T - t)` overflows
/// `Decimal`, or the time to terminal in years does.
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
    let volatility_squared = checked(volatility.checked_mul(volatility), "volatility squared")?;
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
/// Returns `MMError::InvalidConfiguration` if any parameter is not positive.
///
/// Returns `MMError::NumericalError` if:
/// - `σ²`, the inventory risk term `γ * σ² * (T - t)`, `γ / k`, the adverse
///   selection term or the spread overflows `Decimal`
/// - `γ / k` underflows: it rounds to zero at the 28 decimal places of `Decimal`
///   (for example `γ = 1e-20` with `k = 1e10`)
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
    let volatility_squared = checked(volatility.checked_mul(volatility), "volatility squared")?;
    let inventory_risk_term = checked(
        risk_aversion
            .checked_mul(volatility_squared)
            .and_then(|x| x.checked_mul(time_to_terminal_years)),
        "inventory risk term",
    )?;

    // (2/γ) * ln(1 + γ/k) = (2/k) * L, with L = log1p(γ/k) / (γ/k).
    let ratio = checked(risk_aversion.checked_div(order_intensity), "gamma / k")?;
    let (_, log_ratio) = log1p_ratio_parts(ratio)?;
    let adverse_selection_term = checked(
        Decimal::TWO
            .checked_div(order_intensity)
            .and_then(|x| x.checked_mul(log_ratio)),
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
/// Returns the errors of [`calculate_reservation_price`] and
/// [`calculate_optimal_spread`], `MMError::NumericalError` if the half spread,
/// bid or ask overflows, and `MMError::InvalidQuoteGeneration` if the bid is not
/// below the ask or not positive.
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

/// `log1p(x) / x` for a positive `x`, through `libm` and IEEE basic operations only:
/// the unrounded `f64` ratio and the ratio rounded half to even to
/// [`LOG_RATIO_SIGNIFICANT_DIGITS`]. Exposed for the fixture tests.
///
/// # Errors
///
/// `MMError::NumericalError` with "gamma / k underflows" if `x` is not positive.
#[doc(hidden)]
pub fn log1p_ratio_parts(x: Decimal) -> MMResult<(f64, Decimal)> {
    if x <= Decimal::ZERO {
        return Err(MMError::NumericalError(
            "avellaneda-stoikov: gamma / k underflows".to_string(),
        ));
    }
    // `decimal_to_f64` is a correctly rounded parse, so the f64 does not depend on the
    // target. A positive Decimal is at least 1e-28 and below 8e28, so it is a positive,
    // finite f64.
    let xf = decimal_to_f64(x)?;
    let ratio = libm::log1p(xf) / xf;
    // `ratio` lies in (8e-28, 1]; the rounding reads its exact binary value. Below 1e-16
    // (x above about 4e17) 12 significant digits would need more than the 28 decimal
    // places of `Decimal`, so the ratio keeps 28 places there.
    let rounded = if ratio < SIGNIFICANT_ROUNDING_FLOOR {
        round_half_even_decimals(ratio, Decimal::MAX_SCALE)?
    } else {
        round_half_even_significant(ratio, LOG_RATIO_SIGNIFICANT_DIGITS)?
    };
    Ok((ratio, rounded))
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
