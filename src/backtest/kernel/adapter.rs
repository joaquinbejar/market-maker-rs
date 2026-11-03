//! Classic strategies as kernel actions.
//!
//! [`QuoteAdapter`] turns a strategy quote into a [`KernelAction`] exactly
//! as the engine treats a `BacktestStrategy` quote: the crossed check runs
//! on the unrounded quote, then bids round down, asks up and sizes down to
//! the grid (a size that is not positive uses the default order size). A
//! strategy error is a [`KernelAction::QuoteError`], the engine's `None`.
//!
//! Two cases fall outside the equivalence subset: a rounded price outside
//! `1..=max_price` ticks or a rounded size outside `1..=max_order_size` lots
//! becomes a `QuoteError` (the engine would submit it and reject it), and a
//! quote that is crossed before rounding but not after has its sides
//! swapped so it stays crossed (both sides are rejected locally, as in the
//! engine, with the swapped prices in the reports).

use crate::Decimal;
use crate::backtest::engine::round_to_step;
use crate::strategy::avellaneda_stoikov::calculate_optimal_quotes;
use crate::strategy::glft::{GLFTConfig, GLFTStrategy};
use crate::strategy::quote::Quote;
use crate::types::error::{MMError, MMResult};

use super::FinancialKernel;
use super::config::KernelConfig;
use super::grid::Unit;
use super::observe::Observation;
use super::state::{ErrCode, KernelAction};

/// Code of a quote whose rounded price is not a positive tick count within
/// `max_price`.
pub const ERR_PRICE_OUT_OF_RANGE: ErrCode = ErrCode(100);
/// Code of a quote whose rounded size is zero or above `max_order_size`.
pub const ERR_SIZE_OUT_OF_RANGE: ErrCode = ErrCode(101);

/// [`ErrCode`] of a strategy error: the `MMError` discriminant plus one.
#[must_use]
pub fn quote_error_code(error: &MMError) -> ErrCode {
    ErrCode(match error {
        MMError::InvalidConfiguration(_) => 1,
        MMError::InvalidMarketState(_) => 2,
        MMError::NumericalError(_) => 3,
        MMError::InvalidPositionUpdate(_) => 4,
        MMError::InvalidQuoteGeneration(_) => 5,
        MMError::InvalidTimestamp(_) => 6,
        MMError::ConnectionError(_) => 7,
        MMError::UnsupportedAction { .. } => 8,
        MMError::LockPoisoned(_) => 9,
        MMError::UnsupportedContract { .. } => 10,
        MMError::PriceUnderflow => 11,
    })
}

/// A strategy that quotes from an [`Observation`].
pub trait QuoteSource {
    /// Quote at the observed `mid` with the known `inventory`.
    ///
    /// # Errors
    ///
    /// Any strategy error; it becomes a [`KernelAction::QuoteError`].
    fn quote(&mut self, obs: &Observation, mid: Decimal, inventory: Decimal) -> MMResult<Quote>;
}

/// Avellaneda-Stoikov quotes (Avellaneda and Stoikov, 2008, eqs. 29 and
/// 30) through [`calculate_optimal_quotes`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AvellanedaQuoter {
    /// Risk aversion `γ`.
    pub risk_aversion: Decimal,
    /// Volatility `σ`.
    pub volatility: Decimal,
    /// Order intensity `k`.
    pub order_intensity: Decimal,
    /// Terminal time, in milliseconds of the observed rows' clock.
    pub terminal_ms: u64,
    /// Size of both sides.
    pub size: Decimal,
}

impl QuoteSource for AvellanedaQuoter {
    fn quote(&mut self, obs: &Observation, mid: Decimal, inventory: Decimal) -> MMResult<Quote> {
        let now_ms = obs.ts_ns / 1_000_000;
        let remaining = self.terminal_ms.saturating_sub(now_ms);
        let (bid, ask) = calculate_optimal_quotes(
            mid,
            inventory,
            self.risk_aversion,
            self.volatility,
            remaining,
            self.order_intensity,
        )?;
        Ok(Quote {
            bid_price: bid,
            bid_size: self.size,
            ask_price: ask,
            ask_size: self.size,
            timestamp: now_ms,
        })
    }
}

/// GLFT quotes (Guéant, Lehalle and Fernandez-Tapia, arXiv:1105.3115,
/// Theorem 1) through [`GLFTStrategy::calculate_optimal_quotes`]. A side
/// missing at the inventory bound is an error.
#[derive(Debug, Clone, PartialEq)]
pub struct GlftQuoter {
    /// Model configuration.
    pub config: GLFTConfig,
    /// Volatility `σ`.
    pub volatility: Decimal,
    /// Size of both sides.
    pub size: Decimal,
}

impl QuoteSource for GlftQuoter {
    fn quote(&mut self, obs: &Observation, mid: Decimal, inventory: Decimal) -> MMResult<Quote> {
        let now_ms = obs.ts_ns / 1_000_000;
        let quotes = GLFTStrategy::calculate_optimal_quotes(
            mid,
            inventory,
            &self.config,
            self.volatility,
            now_ms,
        )?;
        match (quotes.bid, quotes.ask) {
            (Some(bid), Some(ask)) => Ok(Quote {
                bid_price: bid,
                bid_size: self.size,
                ask_price: ask,
                ask_size: self.size,
                timestamp: now_ms,
            }),
            (None, _) | (_, None) => Err(MMError::InvalidQuoteGeneration(format!(
                "one-sided glft quote at inventory {inventory}"
            ))),
        }
    }
}

/// Drives a [`QuoteSource`] from kernel observations.
#[derive(Debug, Clone)]
pub struct QuoteAdapter<S> {
    source: S,
    tick: Unit,
    lot: Unit,
    max_price: u64,
    max_lots: u64,
    default_order_size: Decimal,
}

impl<S: QuoteSource> QuoteAdapter<S> {
    /// An adapter on the grid of `cfg`; `default_order_size` replaces a
    /// size that is not positive, as the engine's
    /// `BacktestConfig::default_order_size`.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] if the grid of `cfg` is
    /// invalid.
    pub fn new(source: S, cfg: &KernelConfig, default_order_size: Decimal) -> MMResult<Self> {
        let invalid =
            |what: &str| MMError::InvalidConfiguration(format!("adapter: invalid {what}"));
        let tick = Unit::new(cfg.tick_size).ok_or_else(|| invalid("tick_size"))?;
        let lot = Unit::new(cfg.lot_size).ok_or_else(|| invalid("lot_size"))?;
        let max_price = tick
            .count(cfg.max_price)
            .and_then(|v| u64::try_from(v).ok())
            .ok_or_else(|| invalid("max_price"))?;
        let max_lots = lot
            .count(cfg.max_order_size)
            .and_then(|v| u64::try_from(v).ok())
            .ok_or_else(|| invalid("max_order_size"))?;
        Ok(Self {
            source,
            tick,
            lot,
            max_price,
            max_lots,
            default_order_size,
        })
    }

    /// The wrapped strategy.
    #[must_use]
    pub fn source(&self) -> &S {
        &self.source
    }

    /// The action for the observation of `kernel`.
    pub fn act<const ORD: usize, const MSG: usize, const DLV: usize, const OBS: usize>(
        &mut self,
        kernel: &FinancialKernel<'_, ORD, MSG, DLV, OBS>,
        obs: &Observation,
    ) -> KernelAction {
        let mid = kernel
            .price(obs.bid)
            .zip(kernel.price(obs.ask))
            .and_then(|(b, a)| b.checked_add(a))
            .and_then(|sum| sum.checked_div(Decimal::TWO));
        let Some(mid) = mid else {
            return KernelAction::QuoteError(ERR_PRICE_OUT_OF_RANGE);
        };
        match self.source.quote(obs, mid, obs.position) {
            Ok(quote) => self.to_action(&quote),
            Err(error) => KernelAction::QuoteError(quote_error_code(&error)),
        }
    }

    /// Converts a quote as the engine's `replace_quotes` does.
    #[must_use]
    pub fn to_action(&self, quote: &Quote) -> KernelAction {
        let crossed = quote.bid_price >= quote.ask_price;
        let bid = self.ticks(round_to_step(quote.bid_price, self.tick.value, false));
        let ask = self.ticks(round_to_step(quote.ask_price, self.tick.value, true));
        let (Some(bid), Some(ask)) = (bid, ask) else {
            return KernelAction::QuoteError(ERR_PRICE_OUT_OF_RANGE);
        };
        let (Some(bid_lots), Some(ask_lots)) =
            (self.lots(quote.bid_size), self.lots(quote.ask_size))
        else {
            return KernelAction::QuoteError(ERR_SIZE_OUT_OF_RANGE);
        };
        // A crossed quote stays crossed in ticks, so both sides are
        // rejected locally as in the engine.
        let (bid, ask) = if crossed && bid < ask {
            (ask, bid)
        } else {
            (bid, ask)
        };
        KernelAction::Quote {
            bid,
            ask,
            bid_lots,
            ask_lots,
        }
    }

    fn ticks(&self, price: Decimal) -> Option<u64> {
        let ticks = u64::try_from(self.tick.count(price)?).ok()?;
        (ticks >= 1 && ticks <= self.max_price).then_some(ticks)
    }

    fn lots(&self, size: Decimal) -> Option<u64> {
        let requested = if size > Decimal::ZERO {
            size
        } else {
            self.default_order_size
        };
        let rounded = round_to_step(requested, self.lot.value, false);
        let lots = u64::try_from(self.lot.count(rounded)?).ok()?;
        (lots >= 1 && lots <= self.max_lots).then_some(lots)
    }
}
