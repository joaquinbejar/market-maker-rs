//! Chain market maker implementation.
//!
//! This module provides the `ChainMarketMaker` for multi-strike quoting
//! across an entire option chain.

use std::sync::Arc;

use option_chain_orderbook::orderbook::ExpirationOrderBook;
use optionstratlib::OptionStyle;
use rust_decimal_macros::dec;

use crate::Decimal;
use crate::chain::risk::{ChainRiskLimits, ChainRiskManager};
use crate::options::greeks::PortfolioGreeks;
use crate::options::market_maker::{HedgeOrder, OptionsMarketMakerConfig};
use crate::options::pricing::{OptionInputs, OptionPricer, OptionValuation};
use crate::types::decimal::{OrOverflow, saturating_div};
use crate::types::error::{MMError, MMResult};
use rust_decimal::RoundingStrategy;
use rust_decimal::prelude::ToPrimitive;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Configuration for chain market maker.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ChainMarketMakerConfig {
    /// Base spread in basis points.
    pub base_spread_bps: u64,
    /// ATM spread multiplier (spreads are wider at ATM due to gamma).
    pub atm_spread_multiplier: Decimal,
    /// OTM spread multiplier.
    pub otm_spread_multiplier: Decimal,
    /// Maximum delta per strike.
    pub max_delta_per_strike: Decimal,
    /// Maximum total chain delta.
    pub max_chain_delta: Decimal,
    /// Quote size base.
    pub base_quote_size: u64,
    /// Contract multiplier.
    pub contract_multiplier: Decimal,
    /// ATM tolerance for strike detection (as percentage).
    pub atm_tolerance_pct: Decimal,
    /// Minimum spread in basis points.
    pub min_spread_bps: u64,
    /// Maximum spread in basis points.
    pub max_spread_bps: u64,
}

impl Default for ChainMarketMakerConfig {
    fn default() -> Self {
        Self {
            base_spread_bps: 200, // 2%
            atm_spread_multiplier: dec!(1.5),
            otm_spread_multiplier: dec!(1.0),
            max_delta_per_strike: dec!(50.0),
            max_chain_delta: dec!(500.0),
            base_quote_size: 10,
            contract_multiplier: dec!(100.0),
            atm_tolerance_pct: dec!(0.02), // 2%
            min_spread_bps: 50,            // 0.5%
            max_spread_bps: 1000,          // 10%
        }
    }
}

impl ChainMarketMakerConfig {
    /// Creates a new configuration with custom base spread.
    #[must_use]
    pub fn with_base_spread(base_spread_bps: u64) -> Self {
        Self {
            base_spread_bps,
            ..Default::default()
        }
    }

    /// Converts the options market maker config for individual options.
    #[must_use]
    pub fn to_options_config(&self) -> OptionsMarketMakerConfig {
        // A u64 divided by a positive constant above 1 cannot overflow.
        let fraction = |bps: u64| saturating_div(Decimal::from(bps), dec!(10000.0));
        OptionsMarketMakerConfig {
            base_spread_pct: fraction(self.base_spread_bps),
            min_spread_pct: fraction(self.min_spread_bps),
            max_spread_pct: fraction(self.max_spread_bps),
            gamma_spread_multiplier: self.atm_spread_multiplier,
            vega_spread_multiplier: dec!(0.5),
            theta_spread_multiplier: dec!(0.1),
            atm_tolerance: self.atm_tolerance_pct,
            contract_multiplier: self.contract_multiplier,
            put_call_skew_factor: dec!(1.0),
        }
    }
}

/// Quote update for a single option.
///
/// Every price is in the chain's raw price unit (the unit of the strikes and
/// the underlying price of the [`ExpirationOrderBook`]); `theo` is the
/// model value rounded once, half to even, to that unit, and `valuation`
/// keeps the exact model value and Greeks per unit of underlying.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ChainQuoteUpdate {
    /// Strike price.
    pub strike: u64,
    /// Option style (Call or Put).
    pub style: OptionStyle,
    /// Bid price.
    pub bid_price: u64,
    /// Ask price.
    pub ask_price: u64,
    /// Bid size.
    pub bid_size: u64,
    /// Ask size.
    pub ask_size: u64,
    /// Theoretical value, rounded to the price unit (positive).
    pub theo: u64,
    /// Spread in basis points of `theo`, rounded down.
    pub spread_bps: u64,
    /// The model valuation the quote is built from.
    pub valuation: OptionValuation,
}

impl ChainQuoteUpdate {
    /// Creates a new quote update.
    ///
    /// # Errors
    ///
    /// [`MMError::InvalidQuoteGeneration`] when `theo` is zero, the book is
    /// crossed (`bid_price > ask_price`) or the spread in basis points
    /// overflows.
    #[allow(clippy::too_many_arguments)] // one field per quote attribute
    pub fn new(
        strike: u64,
        style: OptionStyle,
        bid_price: u64,
        ask_price: u64,
        bid_size: u64,
        ask_size: u64,
        theo: u64,
        valuation: OptionValuation,
    ) -> MMResult<Self> {
        if theo == 0 {
            return Err(MMError::InvalidQuoteGeneration(format!(
                "theoretical value of strike {strike} is zero"
            )));
        }
        let spread = ask_price.checked_sub(bid_price).ok_or_else(|| {
            MMError::InvalidQuoteGeneration(format!(
                "crossed quote: bid {bid_price} above ask {ask_price}"
            ))
        })?;
        let spread_bps = spread
            .checked_mul(10_000)
            .and_then(|x| x.checked_div(theo))
            .ok_or_else(|| {
                MMError::InvalidQuoteGeneration(format!(
                    "spread {spread} in basis points of theo {theo} overflows"
                ))
            })?;
        Ok(Self {
            strike,
            style,
            bid_price,
            ask_price,
            bid_size,
            ask_size,
            theo,
            spread_bps,
            valuation,
        })
    }

    /// Returns the mid price, rounded down; computed without overflow.
    #[must_use]
    pub fn mid_price(&self) -> u64 {
        self.bid_price.midpoint(self.ask_price)
    }

    /// Returns the spread, `ask - bid`, or `None` for a crossed quote.
    #[must_use]
    pub fn spread(&self) -> Option<u64> {
        self.ask_price.checked_sub(self.bid_price)
    }
}

/// The outcome of one contract in a chain refresh.
#[derive(Debug, Clone)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ChainQuoteOutcome {
    /// A quote to publish.
    Quote(ChainQuoteUpdate),
    /// No quote for this contract: any quote resting from an earlier refresh
    /// should be withdrawn by the caller.
    NotQuoted {
        /// Strike price.
        strike: u64,
        /// Option style (Call or Put).
        style: OptionStyle,
        /// Why the contract is not quoted (for example a theoretical value
        /// below one price unit).
        reason: MMError,
    },
}

/// Valuation inputs shared by every contract of the chain's single
/// expiration.
///
/// `tau_years` is the time from the valuation instant to the expiration of
/// the [`ExpirationOrderBook`], in years of
/// [`crate::options::pricing::PRICING_YEAR_DAYS`] days, supplied by the
/// caller (there is no clock here). The pricer holds the surface and its
/// dynamics; the rates live here only, so they are never duplicated.
#[derive(Clone)]
pub struct ChainPricing {
    pricer: Arc<dyn OptionPricer>,
    tau_years: Decimal,
    rate: Decimal,
    carry: Decimal,
}

impl std::fmt::Debug for ChainPricing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChainPricing")
            .field("tau_years", &self.tau_years)
            .field("rate", &self.rate)
            .field("carry", &self.carry)
            .finish_non_exhaustive()
    }
}

impl ChainPricing {
    /// Shared valuation inputs for one expiration.
    ///
    /// # Errors
    ///
    /// [`MMError::InvalidTimestamp`] when `tau_years` is negative.
    pub fn new(
        pricer: Arc<dyn OptionPricer>,
        tau_years: Decimal,
        rate: Decimal,
        carry: Decimal,
    ) -> MMResult<Self> {
        if tau_years < Decimal::ZERO {
            return Err(MMError::InvalidTimestamp(format!(
                "time to expiry must be non-negative, got {tau_years}"
            )));
        }
        Ok(Self {
            pricer,
            tau_years,
            rate,
            carry,
        })
    }

    /// Time to expiry in years.
    #[must_use]
    pub fn tau_years(&self) -> Decimal {
        self.tau_years
    }

    /// Continuously compounded risk-free rate.
    #[must_use]
    pub fn rate(&self) -> Decimal {
        self.rate
    }

    /// Continuously compounded carry.
    #[must_use]
    pub fn carry(&self) -> Decimal {
        self.carry
    }

    fn inputs(&self, underlying: u64, strike: u64, style: OptionStyle) -> OptionInputs {
        OptionInputs::european(
            Decimal::from(underlying),
            Decimal::from(strike),
            self.tau_years,
            self.rate,
            self.carry,
            style,
        )
    }
}

/// The model price rounded once, half to even, to the chain's price unit.
fn theo_to_price_unit(strike: u64, valuation: &OptionValuation) -> MMResult<u64> {
    let rounded = valuation
        .price
        .round_dp_with_strategy(0, RoundingStrategy::MidpointNearestEven);
    let theo = rounded.to_u64().ok_or_else(|| {
        MMError::InvalidQuoteGeneration(format!(
            "theoretical value {} of strike {strike} is not a price unit count",
            valuation.price
        ))
    })?;
    if theo == 0 {
        return Err(MMError::InvalidQuoteGeneration(format!(
            "theoretical value {} of strike {strike} rounds to zero price units",
            valuation.price
        )));
    }
    Ok(theo)
}

/// Risk status for the chain.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum RiskStatus {
    /// All risk limits are within bounds.
    Normal,
    /// Approaching risk limits (warning).
    Warning,
    /// Risk limits breached, or a fill could not be booked: quoting is refused.
    Breached,
    /// Hedge required: quoting is refused while the delta is above
    /// `delta_hedge_threshold`. Nothing here hedges, so it lasts until the
    /// caller does: execute [`ChainMarketMaker::calculate_hedge`], book the
    /// fill ([`ChainRiskManager::add_position_greeks`]), and quoting resumes.
    HedgeRequired,
}

impl RiskStatus {
    /// Returns true if quoting is allowed. While it is false,
    /// [`ChainMarketMaker::generate_quote`] and
    /// [`ChainMarketMaker::refresh_all_quotes`] refuse with
    /// [`MMError::QuotingHalted`].
    #[must_use]
    pub fn can_quote(&self) -> bool {
        matches!(self, RiskStatus::Normal | RiskStatus::Warning)
    }

    /// Returns true if immediate action is required.
    #[must_use]
    pub fn requires_action(&self) -> bool {
        matches!(self, RiskStatus::Breached | RiskStatus::HedgeRequired)
    }
}

/// Market maker for an entire option chain.
///
/// Provides multi-strike quoting, chain-level Greeks aggregation,
/// and risk management for an option expiration.
pub struct ChainMarketMaker {
    /// The option chain.
    chain: Arc<ExpirationOrderBook>,
    /// Configuration.
    config: ChainMarketMakerConfig,
    /// Risk manager.
    risk_manager: ChainRiskManager,
    /// Underlying symbol.
    underlying_symbol: String,
}

impl ChainMarketMaker {
    /// Creates a new chain market maker.
    ///
    /// # Arguments
    ///
    /// * `chain` - The expiration order book
    /// * `config` - Market maker configuration
    #[must_use]
    pub fn new(chain: Arc<ExpirationOrderBook>, config: ChainMarketMakerConfig) -> Self {
        let underlying_symbol = chain.underlying().to_string();
        let risk_limits = ChainRiskLimits {
            max_chain_delta: config.max_chain_delta,
            max_delta_per_strike: config.max_delta_per_strike,
            ..Default::default()
        };

        Self {
            chain,
            risk_manager: ChainRiskManager::new(
                &underlying_symbol,
                risk_limits,
                config.contract_multiplier,
            ),
            config,
            underlying_symbol,
        }
    }

    /// Creates a new chain market maker with default configuration.
    #[must_use]
    pub fn with_defaults(chain: Arc<ExpirationOrderBook>) -> Self {
        Self::new(chain, ChainMarketMakerConfig::default())
    }

    /// Returns a reference to the chain.
    #[must_use]
    pub fn chain(&self) -> &ExpirationOrderBook {
        &self.chain
    }

    /// Returns a reference to the configuration.
    #[must_use]
    pub fn config(&self) -> &ChainMarketMakerConfig {
        &self.config
    }

    /// Returns a reference to the risk manager.
    #[must_use]
    pub fn risk_manager(&self) -> &ChainRiskManager {
        &self.risk_manager
    }

    /// Returns a mutable reference to the risk manager.
    ///
    /// Replacing the whole manager through it also discards a failed booking
    /// (see [`ChainRiskManager::has_unbooked_fill`]) and, with its Greeks,
    /// whatever kept [`Self::check_chain_risk`] from allowing quotes.
    pub fn risk_manager_mut(&mut self) -> &mut ChainRiskManager {
        &mut self.risk_manager
    }

    /// Returns the underlying symbol.
    #[must_use]
    pub fn underlying_symbol(&self) -> &str {
        &self.underlying_symbol
    }

    /// Gets the ATM strike for the given spot price.
    ///
    /// # Arguments
    ///
    /// * `spot_price` - Current underlying price
    ///
    /// # Returns
    ///
    /// The ATM strike price, or an error if no strikes exist.
    pub fn get_atm_strike(&self, spot_price: u64) -> MMResult<u64> {
        self.chain
            .atm_strike(spot_price)
            .map_err(|e| crate::types::error::MMError::InvalidMarketState(e.to_string()))
    }

    /// Calculates the spread multiplier based on moneyness.
    ///
    /// # Arguments
    ///
    /// * `strike` - Strike price
    /// * `spot_price` - Current underlying price
    /// * `is_atm` - Whether this is the ATM strike
    ///
    /// # Errors
    ///
    /// [`MMError::InvalidMarketState`] for a zero spot price, and
    /// [`MMError::NumericalError`] when the multiplier overflows.
    pub fn get_spread_multiplier(
        &self,
        strike: u64,
        spot_price: u64,
        is_atm: bool,
    ) -> MMResult<Decimal> {
        if is_atm {
            return Ok(self.config.atm_spread_multiplier);
        }
        if spot_price == 0 {
            return Err(MMError::InvalidMarketState(
                "spot price must be positive".to_string(),
            ));
        }
        // Further OTM options get slightly wider spreads.
        let overflow =
            || MMError::NumericalError(format!("spread multiplier of strike {strike} overflows"));
        let moneyness = Decimal::from(strike)
            .checked_div(Decimal::from(spot_price))
            .ok_or_else(overflow)?;
        let otm_factor = moneyness
            .checked_sub(Decimal::ONE)
            .ok_or_else(overflow)?
            .abs();
        otm_factor
            .checked_mul(dec!(0.5))
            .and_then(|x| x.checked_add(self.config.otm_spread_multiplier))
            .ok_or_else(overflow)
    }

    /// Calculates the quote size based on strike characteristics.
    ///
    /// # Arguments
    ///
    /// * `strike` - Strike price
    /// * `spot_price` - Current underlying price
    /// * `is_atm` - Whether this is the ATM strike
    ///
    /// # Errors
    ///
    /// [`MMError::NumericalError`] when the ATM size overflows.
    pub fn calculate_quote_size(
        &self,
        _strike: u64,
        _spot_price: u64,
        is_atm: bool,
    ) -> MMResult<u64> {
        if is_atm {
            // ATM options typically have more liquidity
            self.config.base_quote_size.checked_mul(2).ok_or_else(|| {
                MMError::NumericalError(format!(
                    "ATM quote size {} x 2 overflows",
                    self.config.base_quote_size
                ))
            })
        } else {
            Ok(self.config.base_quote_size)
        }
    }

    /// Generates a quote for a single option from its valuation.
    ///
    /// The theoretical value is the model price rounded once, half to even,
    /// to the chain's price unit; the half spread is
    /// `theo * spread_bps / 20000` rounded down, clamped between the
    /// configured minimum and maximum spreads.
    ///
    /// # Arguments
    ///
    /// * `strike` - Strike price
    /// * `style` - Option style (Call or Put)
    /// * `spot_price` - Current underlying price
    /// * `valuation` - The model valuation of the option
    /// * `is_atm` - Whether this is the ATM strike
    ///
    /// # Errors
    ///
    /// [`MMError::QuotingHalted`] while [`Self::check_chain_risk`] does not
    /// allow quoting, and its error when it cannot be computed;
    /// [`MMError::InvalidQuoteGeneration`] when the theoretical value rounds
    /// to zero or is not a price unit count, or the quote cannot be built;
    /// [`MMError::NumericalError`] when the spread arithmetic overflows.
    pub fn generate_quote(
        &self,
        strike: u64,
        style: OptionStyle,
        spot_price: u64,
        valuation: OptionValuation,
        is_atm: bool,
    ) -> MMResult<ChainQuoteUpdate> {
        self.ensure_can_quote()?;
        self.quote_contract(strike, style, spot_price, valuation, is_atm)
    }

    /// Refuses unless the chain risk status allows quoting.
    fn ensure_can_quote(&self) -> MMResult<()> {
        let status = self.check_chain_risk()?;
        if status.can_quote() {
            Ok(())
        } else {
            Err(MMError::QuotingHalted(format!(
                "chain risk status is {status:?}"
            )))
        }
    }

    /// [`Self::generate_quote`] once the risk status allows quoting.
    fn quote_contract(
        &self,
        strike: u64,
        style: OptionStyle,
        spot_price: u64,
        valuation: OptionValuation,
        is_atm: bool,
    ) -> MMResult<ChainQuoteUpdate> {
        let theo_price = theo_to_price_unit(strike, &valuation)?;
        let spread_mult = self.get_spread_multiplier(strike, spot_price, is_atm)?;
        let overflow = || MMError::NumericalError(format!("spread of strike {strike} overflows"));
        let adjusted_spread_bps = Decimal::from(self.config.base_spread_bps)
            .checked_mul(spread_mult)
            .ok_or_else(overflow)?;
        let (min_bps, max_bps) = (
            Decimal::from(self.config.min_spread_bps),
            Decimal::from(self.config.max_spread_bps),
        );
        let clamped_spread_bps = if adjusted_spread_bps < min_bps {
            min_bps
        } else if adjusted_spread_bps > max_bps {
            max_bps
        } else {
            adjusted_spread_bps
        };
        let half_spread = Decimal::from(theo_price)
            .checked_mul(clamped_spread_bps)
            .and_then(|x| x.checked_div(dec!(20000)))
            .ok_or_else(overflow)?
            .trunc()
            .to_u64()
            .ok_or_else(overflow)?;
        let bid_price = theo_price
            .checked_sub(half_spread)
            .filter(|bid| *bid > 0)
            .ok_or_else(|| {
                MMError::InvalidQuoteGeneration(format!(
                    "half spread {half_spread} of strike {strike} reaches its theo {theo_price}: no positive bid"
                ))
            })?;
        let ask_price = theo_price.checked_add(half_spread).ok_or_else(overflow)?;
        let quote_size = self.calculate_quote_size(strike, spot_price, is_atm)?;
        ChainQuoteUpdate::new(
            strike, style, bid_price, ask_price, quote_size, quote_size, theo_price, valuation,
        )
    }

    /// Refreshes all quotes of the chain from the model.
    ///
    /// Each contract is valued with `pricing`, then quoted. The result has
    /// one outcome per contract (strike x call/put): a quote, or the reason
    /// it is not quoted (for example a theoretical value below one price
    /// unit, or a strike the surface does not cover); a failed contract
    /// never gets a substitute value. A failure of what the contracts share
    /// fails the whole refresh and nothing is returned to publish, so any
    /// resting quote should be withdrawn. That includes a risk status that does
    /// not allow quoting, checked first, before anything is valued.
    ///
    /// # Arguments
    ///
    /// * `underlying_price` - Current price of the underlying, in the
    ///   chain's price unit
    /// * `pricing` - Valuation inputs for this expiration
    ///
    /// # Errors
    ///
    /// [`MMError::QuotingHalted`] while [`Self::check_chain_risk`] does not
    /// allow quoting, and its error when it cannot be computed;
    /// [`MMError::InvalidMarketState`] for a zero underlying price, and the
    /// error of [`OptionPricer::check_market`] for the inputs every contract
    /// shares (spot, rates, time to expiry and the factors derived from
    /// them). The surface is never evaluated for the shared check: a surface
    /// error at a contract's own `k` is that contract's `NotQuoted`.
    pub fn refresh_all_quotes(
        &self,
        underlying_price: u64,
        pricing: &ChainPricing,
    ) -> MMResult<Vec<ChainQuoteOutcome>> {
        self.ensure_can_quote()?;
        if underlying_price == 0 {
            return Err(MMError::InvalidMarketState(
                "underlying price must be positive".to_string(),
            ));
        }
        // The inputs every contract shares, checked without valuing any.
        pricing.pricer.check_market(
            Decimal::from(underlying_price),
            pricing.tau_years,
            pricing.rate,
            pricing.carry,
        )?;
        let atm_strike = self.get_atm_strike(underlying_price).ok();
        let strikes = self.chain.strike_prices();
        let mut outcomes = Vec::with_capacity(strikes.len().saturating_mul(2));
        for strike in strikes {
            let is_atm = atm_strike == Some(strike);
            for style in [OptionStyle::Call, OptionStyle::Put] {
                let quote = pricing
                    .pricer
                    .value(&pricing.inputs(underlying_price, strike, style))
                    .and_then(|valuation| {
                        self.quote_contract(strike, style, underlying_price, valuation, is_atm)
                    });
                outcomes.push(match quote {
                    Ok(q) => ChainQuoteOutcome::Quote(q),
                    Err(reason) => ChainQuoteOutcome::NotQuoted {
                        strike,
                        style,
                        reason,
                    },
                });
            }
        }
        Ok(outcomes)
    }

    /// Gets the aggregated Greeks for the entire chain.
    #[must_use]
    pub fn get_chain_greeks(&self) -> &PortfolioGreeks {
        self.risk_manager.current_greeks()
    }

    /// Checks the chain-level risk status. It is [`RiskStatus::Breached`], so
    /// [`RiskStatus::can_quote`] is false, while the Greeks miss a fill that
    /// could not be booked (see [`ChainRiskManager::has_unbooked_fill`]).
    ///
    /// [`Self::generate_quote`] and [`Self::refresh_all_quotes`] call it first
    /// and refuse while [`RiskStatus::can_quote`] is false.
    ///
    /// # Errors
    ///
    /// [`MMError::NumericalError`] for a zero delta, gamma or vega limit that
    /// is not already breached: the utilization of a zero limit is undefined.
    pub fn check_chain_risk(&self) -> MMResult<RiskStatus> {
        if self.risk_manager.is_risk_breached() {
            Ok(RiskStatus::Breached)
        } else if self.risk_manager.should_hedge()? {
            Ok(RiskStatus::HedgeRequired)
        } else {
            // Check if approaching limits (80% threshold)
            let greeks = self.risk_manager.current_greeks();
            let limits = self.risk_manager.limits();

            let delta_util = greeks
                .delta
                .abs()
                .checked_div(limits.max_chain_delta)
                .or_overflow("chain delta utilization")?;
            let gamma_util = greeks
                .gamma
                .abs()
                .checked_div(limits.max_chain_gamma)
                .or_overflow("chain gamma utilization")?;
            let vega_util = greeks
                .vega
                .abs()
                .checked_div(limits.max_chain_vega)
                .or_overflow("chain vega utilization")?;

            if delta_util > dec!(0.8) || gamma_util > dec!(0.8) || vega_util > dec!(0.8) {
                Ok(RiskStatus::Warning)
            } else {
                Ok(RiskStatus::Normal)
            }
        }
    }

    /// Calculates hedge orders to neutralize chain delta.
    ///
    /// # Arguments
    ///
    /// * `underlying_price` - Current price of the underlying
    ///
    /// # Errors
    ///
    /// [`MMError::GreeksMissFill`] while the Greeks miss a fill that could not
    /// be booked, and [`MMError::NumericalError`] when the number of shares
    /// overflows.
    pub fn calculate_hedge(&self, underlying_price: Decimal) -> MMResult<Vec<HedgeOrder>> {
        self.risk_manager.calculate_hedge(underlying_price)
    }

    /// Returns the number of strikes in the chain.
    #[must_use]
    pub fn strike_count(&self) -> usize {
        self.chain.strike_count()
    }

    /// Returns all strike prices (sorted).
    pub fn strike_prices(&self) -> Vec<u64> {
        self.chain.strike_prices()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use optionstratlib::ExpirationDate;
    use optionstratlib::prelude::pos_or_panic;

    fn create_test_chain() -> Arc<ExpirationOrderBook> {
        let exp = ExpirationDate::Days(pos_or_panic!(30.0));
        let chain = ExpirationOrderBook::new("BTC", exp);

        // Add some strikes
        chain.get_or_create_strike(45000);
        chain.get_or_create_strike(50000);
        chain.get_or_create_strike(55000);

        Arc::new(chain)
    }

    #[test]
    fn test_chain_market_maker_config_default() {
        let config = ChainMarketMakerConfig::default();
        assert_eq!(config.base_spread_bps, 200);
        assert_eq!(config.atm_spread_multiplier, dec!(1.5));
    }

    #[test]
    fn test_chain_market_maker_new() {
        let chain = create_test_chain();
        let mm = ChainMarketMaker::with_defaults(chain);

        assert_eq!(mm.underlying_symbol(), "BTC");
        assert_eq!(mm.strike_count(), 3);
    }

    #[test]
    fn test_get_atm_strike() {
        let chain = create_test_chain();
        let mm = ChainMarketMaker::with_defaults(chain);

        let atm = mm.get_atm_strike(50000).unwrap();
        assert_eq!(atm, 50000);

        let atm = mm.get_atm_strike(48000).unwrap();
        assert_eq!(atm, 50000); // Closest strike
    }

    fn flat_pricing(sigma: Decimal, tau_years: Decimal) -> ChainPricing {
        use crate::options::pricing::{Dynamics, SurfacePricer};
        use crate::options::surface::FlatSurface;
        let pricer = SurfacePricer::new(
            Arc::new(FlatSurface::new(sigma).unwrap()),
            Dynamics::StickyMoneyness,
        )
        .unwrap();
        ChainPricing::new(Arc::new(pricer), tau_years, dec!(0.0), dec!(0.0)).unwrap()
    }

    fn valuation(price: Decimal) -> OptionValuation {
        OptionValuation {
            price,
            delta: dec!(0.5),
            gamma: dec!(0.0001),
            vega: dec!(40),
            theta: dec!(-20),
        }
    }

    #[test]
    fn test_get_spread_multiplier() {
        let chain = create_test_chain();
        let mm = ChainMarketMaker::with_defaults(chain);

        // ATM should have higher multiplier
        let atm_mult = mm.get_spread_multiplier(50000, 50000, true).unwrap();
        let otm_mult = mm.get_spread_multiplier(55000, 50000, false).unwrap();

        assert!(atm_mult > otm_mult);
        assert!(matches!(
            mm.get_spread_multiplier(55000, 0, false),
            Err(MMError::InvalidMarketState(_))
        ));
    }

    #[test]
    fn test_generate_quote() {
        let chain = create_test_chain();
        let mm = ChainMarketMaker::with_defaults(chain);

        let quote = mm
            .generate_quote(
                50000,
                OptionStyle::Call,
                50000,
                valuation(dec!(1000.4)),
                true,
            )
            .unwrap();

        assert_eq!(quote.strike, 50000);
        assert_eq!(quote.style, OptionStyle::Call);
        assert_eq!(quote.theo, 1000);
        assert_eq!(quote.valuation, valuation(dec!(1000.4)));
        assert!(quote.bid_price < quote.ask_price);
        assert!(quote.bid_size > 0);
    }

    #[test]
    fn test_generate_quote_theo_rounds_half_to_even_once() {
        let mm = ChainMarketMaker::with_defaults(create_test_chain());
        for (price, theo) in [
            (dec!(1000.5), 1000),
            (dec!(1001.5), 1002),
            (dec!(0.5000001), 1),
        ] {
            let q = mm
                .generate_quote(50000, OptionStyle::Put, 50000, valuation(price), false)
                .unwrap();
            assert_eq!(q.theo, theo, "{price}");
        }
    }

    #[test]
    fn test_generate_quote_theo_below_one_unit_is_an_error_not_a_quote() {
        let mm = ChainMarketMaker::with_defaults(create_test_chain());
        for price in [dec!(0.5), dec!(0.0001), dec!(0), dec!(-1)] {
            assert!(
                matches!(
                    mm.generate_quote(50000, OptionStyle::Call, 50000, valuation(price), false),
                    Err(MMError::InvalidQuoteGeneration(_))
                ),
                "{price}"
            );
        }
    }

    #[test]
    fn test_refresh_all_quotes() {
        use crate::options::pricing::{Dynamics, value};
        use crate::options::surface::FlatSurface;
        let chain = create_test_chain();
        let mm = ChainMarketMaker::with_defaults(chain);
        let pricing = flat_pricing(dec!(0.6), dec!(30) / dec!(365));

        let outcomes = mm.refresh_all_quotes(50000, &pricing).unwrap();

        // 3 strikes * 2 styles = 6 outcomes, all quoted.
        assert_eq!(outcomes.len(), 6);
        let surface = FlatSurface::new(dec!(0.6)).unwrap();
        for outcome in &outcomes {
            let ChainQuoteOutcome::Quote(quote) = outcome else {
                panic!("not quoted: {outcome:?}");
            };
            assert!(quote.bid_price < quote.ask_price);
            // theo is the model price, rounded once.
            let inputs = pricing.inputs(50000, quote.strike, quote.style);
            let model = value(&inputs, &surface, Dynamics::StickyMoneyness).unwrap();
            assert_eq!(model, quote.valuation);
            assert_eq!(
                Decimal::from(quote.theo),
                model
                    .price
                    .round_dp_with_strategy(0, RoundingStrategy::MidpointNearestEven)
            );
        }
    }

    #[test]
    fn test_refresh_all_quotes_one_failed_contract_does_not_stop_the_others() {
        let exp = ExpirationDate::Days(pos_or_panic!(30.0));
        let chain = ExpirationOrderBook::new("BTC", exp);
        chain.get_or_create_strike(50000);
        // Far out of the money: the call is worth less than one price unit.
        chain.get_or_create_strike(500_000);
        let mm = ChainMarketMaker::with_defaults(Arc::new(chain));
        let pricing = flat_pricing(dec!(0.6), dec!(30) / dec!(365));

        let outcomes = mm.refresh_all_quotes(50000, &pricing).unwrap();

        assert_eq!(outcomes.len(), 4);
        let mut not_quoted = Vec::new();
        for outcome in &outcomes {
            match outcome {
                ChainQuoteOutcome::Quote(q) => assert!(q.theo > 0),
                ChainQuoteOutcome::NotQuoted {
                    strike,
                    style,
                    reason,
                } => {
                    assert!(matches!(reason, MMError::InvalidQuoteGeneration(_)));
                    not_quoted.push((*strike, *style));
                }
            }
        }
        assert_eq!(not_quoted, vec![(500_000, OptionStyle::Call)]);
    }

    #[test]
    fn test_refresh_all_quotes_shared_failure_fails_the_whole_refresh() {
        use crate::options::pricing::OptionInputs;
        struct NoMarket;
        impl OptionPricer for NoMarket {
            fn value(&self, _: &OptionInputs) -> MMResult<OptionValuation> {
                Ok(valuation(dec!(1000)))
            }
            fn check_market(&self, _: Decimal, _: Decimal, _: Decimal, _: Decimal) -> MMResult<()> {
                Err(MMError::NumericalError("surface unavailable".to_string()))
            }
        }
        let mm = ChainMarketMaker::with_defaults(create_test_chain());
        let no_market = ChainPricing::new(Arc::new(NoMarket), dec!(0.1), dec!(0), dec!(0)).unwrap();
        assert!(mm.refresh_all_quotes(50000, &no_market).is_err());
        let pricing = flat_pricing(dec!(0.6), dec!(0.1));
        assert!(matches!(
            mm.refresh_all_quotes(0, &pricing),
            Err(MMError::InvalidMarketState(_))
        ));
        assert!(ChainPricing::new(Arc::new(NoMarket), dec!(-0.1), dec!(0), dec!(0)).is_err());
        // A surface error is never shared: a surface that answers nowhere
        // leaves every contract NotQuoted with its own error.
        struct NoExpiry;
        impl crate::options::surface::VolSurface for NoExpiry {
            fn total_variance_at(
                &self,
                _: f64,
                _: f64,
            ) -> MMResult<crate::options::surface::TotalVariance> {
                Err(MMError::InvalidMarketState(
                    "tau outside the slices".to_string(),
                ))
            }
        }
        let pricer = crate::options::pricing::SurfacePricer::new(
            Arc::new(NoExpiry),
            crate::options::pricing::Dynamics::StickyMoneyness,
        )
        .unwrap();
        let pricing = ChainPricing::new(Arc::new(pricer), dec!(0.1), dec!(0), dec!(0)).unwrap();
        let outcomes = mm.refresh_all_quotes(50000, &pricing).unwrap();
        assert!(!outcomes.is_empty());
        assert!(
            outcomes.iter().all(|o| matches!(
                o,
                ChainQuoteOutcome::NotQuoted {
                    reason: MMError::InvalidMarketState(_),
                    ..
                }
            )),
            "{outcomes:?}"
        );
        // Rates whose discount factor underflows are shared: the refresh fails.
        let pricing = flat_pricing(dec!(0.6), dec!(1));
        let pricing =
            ChainPricing::new(Arc::clone(&pricing.pricer), dec!(1), dec!(1000), dec!(0)).unwrap();
        assert!(matches!(
            mm.refresh_all_quotes(50000, &pricing),
            Err(MMError::NumericalError(_))
        ));
    }

    #[test]
    fn test_refresh_all_quotes_surface_error_at_k_zero_only_is_not_shared() {
        use crate::options::pricing::{Dynamics, SurfacePricer};
        use crate::options::surface::{TotalVariance, VolSurface};
        // A valid surface that refuses exactly k = 0 (the forward of a
        // strike the chain does not list) and answers at every other k.
        struct NoForwardPoint;
        impl VolSurface for NoForwardPoint {
            fn total_variance_at(&self, k: f64, tau: f64) -> MMResult<TotalVariance> {
                if k == 0.0 {
                    return Err(MMError::InvalidMarketState("k = 0".to_string()));
                }
                TotalVariance::new(0.09 * tau, 0.0, 0.0, 0.09)
            }
        }
        let pricer =
            SurfacePricer::new(Arc::new(NoForwardPoint), Dynamics::StickyMoneyness).unwrap();
        let exp = ExpirationDate::Days(pos_or_panic!(30.0));
        let chain = ExpirationOrderBook::new("X", exp);
        chain.get_or_create_strike(90);
        chain.get_or_create_strike(110);
        let mm = ChainMarketMaker::with_defaults(Arc::new(chain));
        let pricing = ChainPricing::new(Arc::new(pricer), dec!(0.25), dec!(0), dec!(0)).unwrap();
        let outcomes = mm.refresh_all_quotes(100, &pricing).unwrap();
        assert_eq!(outcomes.len(), 4);
        assert!(
            outcomes
                .iter()
                .all(|o| matches!(o, ChainQuoteOutcome::Quote(_))),
            "{outcomes:?}"
        );
    }

    #[test]
    fn test_refresh_all_quotes_does_not_value_a_contract_outside_the_chain() {
        use crate::options::pricing::{Dynamics, OptionInputs, SurfacePricer};
        use crate::options::surface::FlatSurface;
        // A pricer that refuses only spot == strike (a strike the chain
        // below does not list) and values everything else.
        struct NoAtm(SurfacePricer);
        impl OptionPricer for NoAtm {
            fn value(&self, inputs: &OptionInputs) -> MMResult<OptionValuation> {
                if inputs.spot == inputs.strike {
                    return Err(MMError::InvalidMarketState("spot == strike".to_string()));
                }
                self.0.value(inputs)
            }
        }
        let flat = SurfacePricer::new(
            Arc::new(FlatSurface::new(dec!(0.3)).unwrap()),
            Dynamics::StickyMoneyness,
        )
        .unwrap();
        let exp = ExpirationDate::Days(pos_or_panic!(30.0));
        let chain = ExpirationOrderBook::new("X", exp);
        chain.get_or_create_strike(90);
        chain.get_or_create_strike(110);
        let mm = ChainMarketMaker::with_defaults(Arc::new(chain));
        let pricing =
            ChainPricing::new(Arc::new(NoAtm(flat)), dec!(0.25), dec!(0), dec!(0)).unwrap();
        let outcomes = mm.refresh_all_quotes(100, &pricing).unwrap();
        assert_eq!(outcomes.len(), 4);
        assert!(
            outcomes
                .iter()
                .all(|o| matches!(o, ChainQuoteOutcome::Quote(_))),
            "{outcomes:?}"
        );
    }

    #[test]
    fn test_generate_quote_spread_reaching_theo_is_an_error_not_a_zero_bid() {
        let config = ChainMarketMakerConfig {
            base_spread_bps: 30_000,
            max_spread_bps: 30_000,
            ..Default::default()
        };
        let mm = ChainMarketMaker::new(create_test_chain(), config);
        assert!(matches!(
            mm.generate_quote(
                50000,
                OptionStyle::Call,
                50000,
                valuation(dec!(1000)),
                false
            ),
            Err(MMError::InvalidQuoteGeneration(_))
        ));
    }

    #[test]
    fn test_check_chain_risk_normal() {
        let chain = create_test_chain();
        let mm = ChainMarketMaker::with_defaults(chain);

        let status = mm.check_chain_risk().unwrap();
        assert_eq!(status, RiskStatus::Normal);
        assert!(status.can_quote());
    }

    /// Asserts that both quoting methods refuse, the same way, before anything
    /// is valued.
    fn assert_quoting_refused(mm: &ChainMarketMaker, expected: &MMError) {
        use crate::options::pricing::OptionInputs;
        use std::sync::atomic::{AtomicUsize, Ordering};
        struct Counting(Arc<AtomicUsize>);
        impl OptionPricer for Counting {
            fn value(&self, _: &OptionInputs) -> MMResult<OptionValuation> {
                self.0.fetch_add(1, Ordering::SeqCst);
                Ok(valuation(dec!(1000)))
            }
            fn check_market(&self, _: Decimal, _: Decimal, _: Decimal, _: Decimal) -> MMResult<()> {
                self.0.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
        }
        let calls = Arc::new(AtomicUsize::new(0));
        let pricing = ChainPricing::new(
            Arc::new(Counting(Arc::clone(&calls))),
            dec!(0.1),
            dec!(0),
            dec!(0),
        )
        .unwrap();
        assert_eq!(
            mm.refresh_all_quotes(50000, &pricing).as_ref().err(),
            Some(expected)
        );
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert_eq!(
            mm.generate_quote(50000, OptionStyle::Call, 50000, valuation(dec!(1000)), true)
                .as_ref()
                .err(),
            Some(expected)
        );
    }

    fn assert_quoting_allowed(mm: &ChainMarketMaker) {
        let pricing = flat_pricing(dec!(0.6), dec!(30) / dec!(365));
        let outcomes = mm.refresh_all_quotes(50000, &pricing).unwrap();
        assert_eq!(outcomes.len(), 6);
        assert!(
            outcomes
                .iter()
                .all(|o| matches!(o, ChainQuoteOutcome::Quote(_))),
            "{outcomes:?}"
        );
        mm.generate_quote(50000, OptionStyle::Call, 50000, valuation(dec!(1000)), true)
            .unwrap();
    }

    fn halted(status: &str) -> MMError {
        MMError::QuotingHalted(format!("chain risk status is {status}"))
    }

    fn with_greeks(greeks: PortfolioGreeks) -> ChainMarketMaker {
        let mut mm = ChainMarketMaker::with_defaults(create_test_chain());
        mm.risk_manager_mut().update_greeks(greeks);
        mm
    }

    #[test]
    fn test_quoting_is_allowed_while_normal_or_warning() {
        let mm = ChainMarketMaker::with_defaults(create_test_chain());
        assert_eq!(mm.check_chain_risk(), Ok(RiskStatus::Normal));
        assert_quoting_allowed(&mm);

        let mm = with_greeks(PortfolioGreeks {
            gamma: ChainRiskLimits::default().max_chain_gamma * dec!(0.9),
            ..PortfolioGreeks::new()
        });
        assert_eq!(mm.check_chain_risk(), Ok(RiskStatus::Warning));
        assert_quoting_allowed(&mm);
    }

    #[test]
    fn test_quoting_is_refused_while_a_limit_is_breached() {
        let mm = with_greeks(PortfolioGreeks {
            delta: ChainRiskLimits::default().max_chain_delta * dec!(2),
            ..PortfolioGreeks::new()
        });
        assert_eq!(mm.check_chain_risk(), Ok(RiskStatus::Breached));
        assert_quoting_refused(&mm, &halted("Breached"));
    }

    #[test]
    fn test_quoting_is_refused_while_a_hedge_is_required() {
        let limits = ChainRiskLimits::default();
        let mm = with_greeks(PortfolioGreeks {
            delta: limits.delta_hedge_threshold * dec!(2),
            ..PortfolioGreeks::new()
        });
        assert_eq!(mm.check_chain_risk(), Ok(RiskStatus::HedgeRequired));
        assert_quoting_refused(&mm, &halted("HedgeRequired"));

        // Reconciling clears a failed booking, not a required hedge.
        let mut mm = mm;
        let unhedged = PortfolioGreeks {
            delta: limits.delta_hedge_threshold * dec!(2),
            ..PortfolioGreeks::new()
        };
        mm.risk_manager_mut().reconcile(unhedged);
        assert_quoting_refused(&mm, &halted("HedgeRequired"));

        // Once the caller executes the hedge and books it, quotes come back.
        // A share has a delta of one over the contract multiplier.
        let hedges = mm.calculate_hedge(dec!(50000)).unwrap();
        let [hedge] = hedges.as_slice() else {
            panic!("expected one hedge: {hedges:?}");
        };
        let share_delta = -unhedged.delta / hedge.quantity;
        let share = crate::options::greeks::PositionGreeks::new(
            share_delta,
            dec!(0),
            dec!(0),
            dec!(0),
            dec!(0),
        );
        mm.risk_manager_mut()
            .add_position_greeks(&share, hedge.quantity)
            .unwrap();
        assert_eq!(mm.check_chain_risk(), Ok(RiskStatus::Normal));
        assert_quoting_allowed(&mm);
    }

    #[test]
    fn test_quoting_is_refused_while_a_failed_booking_is_latched() {
        let mut mm = ChainMarketMaker::with_defaults(create_test_chain());
        let huge = crate::options::greeks::PositionGreeks::new(
            Decimal::MAX,
            dec!(0),
            dec!(0),
            dec!(0),
            dec!(0),
        );
        assert_quoting_allowed(&mm);
        mm.risk_manager_mut()
            .add_position_greeks(&huge, dec!(2))
            .unwrap_err();
        assert_quoting_refused(&mm, &halted("Breached"));

        // A plain update does not bring quotes back; reconciling does.
        mm.risk_manager_mut().update_greeks(PortfolioGreeks::new());
        assert_quoting_refused(&mm, &halted("Breached"));
        mm.risk_manager_mut().reconcile(PortfolioGreeks::new());
        assert_quoting_allowed(&mm);
    }

    #[test]
    fn test_quoting_is_refused_when_the_status_cannot_be_computed() {
        let mm = with_limits(ChainRiskLimits {
            max_chain_delta: Decimal::ZERO,
            ..Default::default()
        });
        let error = mm.check_chain_risk().unwrap_err();
        assert!(error.is_numerical_error());
        assert_quoting_refused(&mm, &error);
    }

    #[test]
    fn test_refresh_refusal_comes_before_the_inputs_and_contract_errors_stay_per_contract() {
        // The status comes first, whatever the inputs.
        let mm = with_greeks(PortfolioGreeks {
            delta: dec!(1000),
            ..PortfolioGreeks::new()
        });
        let pricing = flat_pricing(dec!(0.6), dec!(0.1));
        assert_eq!(
            mm.refresh_all_quotes(0, &pricing).err(),
            Some(halted("Breached"))
        );

        // And it is the refusal, not a list of NotQuoted, even when no contract
        // can be valued.
        use crate::options::pricing::OptionInputs;
        struct NoValue;
        impl OptionPricer for NoValue {
            fn value(&self, _: &OptionInputs) -> MMResult<OptionValuation> {
                Err(MMError::InvalidMarketState("no value".to_string()))
            }
        }
        let pricing = ChainPricing::new(Arc::new(NoValue), dec!(0.1), dec!(0), dec!(0)).unwrap();
        assert_eq!(
            mm.refresh_all_quotes(50000, &pricing).err(),
            Some(halted("Breached"))
        );

        // While it allows quoting, a contract's own error is still NotQuoted.
        let mm = ChainMarketMaker::with_defaults(create_test_chain());
        let outcomes = mm.refresh_all_quotes(50000, &pricing).unwrap();
        assert_eq!(outcomes.len(), 6);
        assert!(
            outcomes
                .iter()
                .all(|o| matches!(o, ChainQuoteOutcome::NotQuoted { .. }))
        );
    }

    /// A chain market maker with the given chain risk limits.
    fn with_limits(limits: ChainRiskLimits) -> ChainMarketMaker {
        let mut mm = ChainMarketMaker::with_defaults(create_test_chain());
        *mm.risk_manager_mut() = ChainRiskManager::new("BTC", limits, dec!(100.0));
        mm
    }

    #[test]
    fn test_check_chain_risk_zero_limit_is_an_error_not_a_panic() {
        for (name, limits) in [
            (
                "delta",
                ChainRiskLimits {
                    max_chain_delta: Decimal::ZERO,
                    ..Default::default()
                },
            ),
            (
                "gamma",
                ChainRiskLimits {
                    max_chain_gamma: Decimal::ZERO,
                    ..Default::default()
                },
            ),
            (
                "vega",
                ChainRiskLimits {
                    max_chain_vega: Decimal::ZERO,
                    ..Default::default()
                },
            ),
        ] {
            assert_eq!(
                with_limits(limits).check_chain_risk(),
                Err(MMError::NumericalError(format!(
                    "chain {name} utilization overflows"
                )))
            );
        }
    }

    #[test]
    fn test_check_chain_risk_stops_quoting_after_a_failed_booking() {
        let mut mm = with_limits(ChainRiskLimits::default());
        let huge = crate::options::greeks::PositionGreeks::new(
            Decimal::MAX,
            dec!(0),
            dec!(0),
            dec!(0),
            dec!(0),
        );
        assert_eq!(mm.check_chain_risk(), Ok(RiskStatus::Normal));

        assert!(matches!(
            mm.risk_manager_mut().add_position_greeks(&huge, dec!(2)),
            Err(MMError::FillNotBooked(_))
        ));
        let status = mm.check_chain_risk().unwrap();
        assert_eq!(status, RiskStatus::Breached);
        assert!(!status.can_quote());
        assert_eq!(
            mm.calculate_hedge(dec!(100)).map(|hedges| hedges.len()),
            Err(MMError::GreeksMissFill)
        );

        mm.risk_manager_mut().update_greeks(PortfolioGreeks::new());
        assert_eq!(mm.check_chain_risk(), Ok(RiskStatus::Breached));

        mm.risk_manager_mut().reconcile(PortfolioGreeks::new());
        assert_eq!(mm.check_chain_risk(), Ok(RiskStatus::Normal));
    }

    #[test]
    fn test_check_chain_risk_breach_is_reported_before_a_zero_limit() {
        let mut mm = with_limits(ChainRiskLimits {
            max_chain_delta: Decimal::ZERO,
            ..Default::default()
        });
        mm.risk_manager_mut().update_greeks(PortfolioGreeks {
            delta: dec!(1),
            ..PortfolioGreeks::new()
        });
        assert_eq!(mm.check_chain_risk(), Ok(RiskStatus::Breached));
    }

    #[test]
    fn test_calculate_hedge_propagates_overflow() {
        let mut mm = ChainMarketMaker::with_defaults(create_test_chain());
        mm.risk_manager_mut().update_greeks(PortfolioGreeks {
            delta: Decimal::MAX,
            ..PortfolioGreeks::new()
        });
        assert_eq!(
            mm.calculate_hedge(dec!(50000)).map(|hedges| hedges.len()),
            Err(MMError::NumericalError(
                "shares to hedge overflows".to_string()
            ))
        );
    }

    #[test]
    fn test_to_options_config_at_the_bound() {
        let config = ChainMarketMakerConfig {
            base_spread_bps: 200,
            min_spread_bps: 0,
            max_spread_bps: u64::MAX,
            ..Default::default()
        }
        .to_options_config();
        assert_eq!(config.base_spread_pct, dec!(0.02));
        assert_eq!(config.min_spread_pct, Decimal::ZERO);
        assert_eq!(config.max_spread_pct, Decimal::from(u64::MAX) / dec!(10000));
    }

    #[test]
    fn test_risk_status_can_quote() {
        assert!(RiskStatus::Normal.can_quote());
        assert!(RiskStatus::Warning.can_quote());
        assert!(!RiskStatus::Breached.can_quote());
        assert!(!RiskStatus::HedgeRequired.can_quote());
    }

    #[test]
    fn test_chain_quote_update() {
        let quote = ChainQuoteUpdate::new(
            50000,
            OptionStyle::Call,
            980,
            1020,
            10,
            10,
            1000,
            valuation(dec!(1000)),
        )
        .unwrap();

        assert_eq!(quote.mid_price(), 1000);
        assert_eq!(quote.spread(), Some(40));
        assert_eq!(quote.spread_bps, 400); // 4%
    }

    #[test]
    fn test_chain_quote_update_invalid_quotes_are_errors() {
        let v = valuation(dec!(1000));
        let new =
            |bid, ask, theo| ChainQuoteUpdate::new(1, OptionStyle::Call, bid, ask, 1, 1, theo, v);
        assert!(matches!(
            new(1020, 980, 1000),
            Err(MMError::InvalidQuoteGeneration(_))
        ));
        assert!(matches!(
            new(980, 1020, 0),
            Err(MMError::InvalidQuoteGeneration(_))
        ));
        assert!(matches!(
            new(0, u64::MAX, 1),
            Err(MMError::InvalidQuoteGeneration(_))
        ));
    }

    #[test]
    fn test_chain_quote_update_mid_price_does_not_overflow() {
        let mut quote = ChainQuoteUpdate::new(
            1,
            OptionStyle::Put,
            980,
            1020,
            1,
            1,
            1000,
            valuation(dec!(1000)),
        )
        .unwrap();
        quote.bid_price = u64::MAX - 2;
        quote.ask_price = u64::MAX;
        assert_eq!(quote.mid_price(), u64::MAX - 1);
        quote.bid_price = u64::MAX;
        quote.ask_price = 0;
        assert_eq!(quote.spread(), None);
    }
}
