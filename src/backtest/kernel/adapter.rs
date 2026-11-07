//! Classic strategies as kernel actions.
//!
//! [`QuoteAdapter`] turns a strategy quote into a [`KernelAction`] exactly
//! as the engine treats a `BacktestStrategy` quote: the crossed check runs
//! on the unrounded quote, then bids round down, asks up and sizes down to
//! the grid (a size that is not positive uses the default order size). A
//! strategy error is a [`KernelAction::QuoteError`], the engine's `None`.
//! The adapter is bound to one kernel: it takes the grid from it and
//! refuses to act for another.
//!
//! Outside the equivalence subset:
//! - a strategy whose `BacktestStrategy::cancels_when_silent` is true, as nothing here
//!   cancels on silence;
//!
//! - a rounded price outside `1..=max_price` ticks or a rounded size outside
//!   `1..=max_order_size` lots becomes a `QuoteError` (the engine would
//!   submit it and reject it);
//! - a quote that is crossed before rounding but not after has its sides
//!   swapped so it stays crossed (both sides are rejected locally, as in the
//!   engine, with the swapped prices in the reports);
//! - a one-sided quote (GLFT at an inventory bound) becomes a
//!   [`KernelAction::OneSided`]: the engine's quote is always two-sided.

use crate::Decimal;
use crate::backtest::engine::round_to_step;
use crate::strategy::avellaneda_stoikov::calculate_optimal_quotes;
use crate::strategy::glft::{ExactQuotes, GLFTConfig, GLFTStrategy, GlftQuotes, over_cap_error};
use crate::strategy::quote::Quote;
use crate::types::error::{MMError, MMResult};

use super::FinancialKernel;
use super::grid::Unit;
use super::observe::Observation;
use super::state::{ErrCode, KernelAction, QuoteSide};

/// Code of a quote whose rounded price is not a positive tick count within
/// `max_price`.
pub const ERR_PRICE_OUT_OF_RANGE: ErrCode = ErrCode(100);
/// Code of a quote whose rounded size is zero or above `max_order_size`.
pub const ERR_SIZE_OUT_OF_RANGE: ErrCode = ErrCode(101);
/// Code of an adapter asked to act for a kernel it was not built for.
pub const ERR_KERNEL_MISMATCH: ErrCode = ErrCode(102);

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
        MMError::FillNotBooked(_) => 12,
        MMError::FillBookedCheckFailed(_) => 13,
        MMError::GreeksMissFill => 14,
        MMError::QuotingHalted(_) => 15,
    })
}

/// The name of the `MMError` variant a strategy error's [`ErrCode`] stands for, the
/// inverse of [`quote_error_code`]; `None` for a code that is not a strategy error's, such
/// as [`ERR_PRICE_OUT_OF_RANGE`].
#[must_use]
pub fn quote_error_name(code: ErrCode) -> Option<&'static str> {
    Some(match code.0 {
        1 => "InvalidConfiguration",
        2 => "InvalidMarketState",
        3 => "NumericalError",
        4 => "InvalidPositionUpdate",
        5 => "InvalidQuoteGeneration",
        6 => "InvalidTimestamp",
        7 => "ConnectionError",
        8 => "UnsupportedAction",
        9 => "LockPoisoned",
        10 => "UnsupportedContract",
        11 => "PriceUnderflow",
        12 => "FillNotBooked",
        13 => "FillBookedCheckFailed",
        14 => "GreeksMissFill",
        15 => "QuotingHalted",
        _ => return None,
    })
}

/// A strategy quote: both sides, or one side only.
#[derive(Debug, Clone, PartialEq)]
pub enum SourceQuote {
    /// A two-sided quote, as the engine's `BacktestStrategy` returns.
    Both(Quote),
    /// Only a bid: `(price, size)`.
    Bid(Decimal, Decimal),
    /// Only an ask: `(price, size)`.
    Ask(Decimal, Decimal),
}

/// A strategy that quotes from an [`Observation`].
pub trait QuoteSource {
    /// Quote at the observed `mid` with the known `inventory`.
    ///
    /// # Errors
    ///
    /// Any strategy error; it becomes a [`KernelAction::QuoteError`].
    fn quote(
        &mut self,
        obs: &Observation,
        mid: Decimal,
        inventory: Decimal,
    ) -> MMResult<SourceQuote>;

    /// Checks the strategy against the kernel grid (`tick`, `lot`) when the
    /// adapter is built.
    ///
    /// # Errors
    ///
    /// [`MMError::InvalidConfiguration`] on a mismatch.
    fn validate(&self, _tick: Decimal, _lot: Decimal) -> MMResult<()> {
        Ok(())
    }
}

/// Milliseconds since `session_start_ms` at the observation.
fn elapsed_ms(obs: &Observation, session_start_ms: u64) -> MMResult<u64> {
    let now_ms = obs.ts_ns / 1_000_000;
    now_ms.checked_sub(session_start_ms).ok_or_else(|| {
        MMError::InvalidTimestamp(format!(
            "observation at {now_ms} ms precedes the session start {session_start_ms} ms"
        ))
    })
}

/// Avellaneda-Stoikov quotes (Avellaneda and Stoikov, 2008, eqs. 29 and
/// 30) through [`calculate_optimal_quotes`], over a horizon measured from
/// the session start.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AvellanedaQuoter {
    /// Risk aversion `γ`.
    pub risk_aversion: Decimal,
    /// Volatility `σ`.
    pub volatility: Decimal,
    /// Order intensity `k`.
    pub order_intensity: Decimal,
    /// Session start, in milliseconds of the rows' clock.
    pub session_start_ms: u64,
    /// Horizon `T` from the session start, in milliseconds.
    pub horizon_ms: u64,
    /// Size of both sides.
    pub size: Decimal,
}

impl QuoteSource for AvellanedaQuoter {
    fn quote(
        &mut self,
        obs: &Observation,
        mid: Decimal,
        inventory: Decimal,
    ) -> MMResult<SourceQuote> {
        let elapsed = elapsed_ms(obs, self.session_start_ms)?;
        let remaining = self.horizon_ms.checked_sub(elapsed).ok_or_else(|| {
            MMError::InvalidTimestamp(format!(
                "{elapsed} ms since the session start is past the horizon {} ms",
                self.horizon_ms
            ))
        })?;
        let (bid, ask) = calculate_optimal_quotes(
            mid,
            inventory,
            self.risk_aversion,
            self.volatility,
            remaining,
            self.order_intensity,
        )?;
        Ok(SourceQuote::Both(Quote {
            bid_price: bid,
            bid_size: self.size,
            ask_price: ask,
            ask_size: self.size,
            timestamp: obs.ts_ns / 1_000_000,
        }))
    }
}

/// Which GLFT quotes a [`GlftQuoter`] gives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum GlftForm {
    /// The exact quotes of Theorem 1, [`GLFTStrategy::exact_quotes`]. Where the rest of
    /// the horizon is over the matrix exponential's step cap there is none, and the quote
    /// is a strategy error, `NumericalError`: the decision holds.
    #[default]
    Exact,
    /// The closed form approximation, [`GLFTStrategy::closed_form_quotes`], at every
    /// decision. It does not depend on the time left.
    ClosedForm,
    /// The exact quotes, and the closed form only where the exact quote is over the step
    /// cap; any other error of the exact quote stays an error.
    ExactOrClosedForm,
}

/// Which form gave a quote.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum GlftQuoteForm {
    /// The exact quotes of Theorem 1.
    Exact,
    /// The closed form approximation.
    ClosedForm,
}

/// What a [`GlftQuoter`] has done, so a run can tell a strategy that holds from one that
/// fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct GlftCounts {
    /// Quotes from the exact form.
    pub exact: u64,
    /// Quotes from the closed form.
    pub closed_form: u64,
    /// Decisions where the exact quote was over the step cap, under either form that
    /// asks for it.
    pub over_step_cap: u64,
    /// The form of the last quote given, if any.
    pub last: Option<GlftQuoteForm>,
}

/// GLFT quotes (Guéant, Lehalle and Fernandez-Tapia, arXiv:1105.3115,
/// Theorem 1, and its closed form approximation), in the [`GlftForm`] the caller
/// states, with the model time measured from the session start. At an inventory bound
/// the model posts one side only, which becomes a one-sided action. Which form gave each
/// quote is in [`GlftQuoter::counts`]: nothing is substituted silently.
#[derive(Debug, Clone, PartialEq)]
pub struct GlftQuoter {
    /// Model configuration; its `lot_size` must equal the kernel lot.
    pub config: GLFTConfig,
    /// Volatility `σ`.
    pub volatility: Decimal,
    /// Order size; one model lot.
    pub size: Decimal,
    /// Session start (model time 0), in milliseconds of the rows' clock.
    pub session_start_ms: u64,
    /// Which quotes it gives.
    pub form: GlftForm,
    /// What it has done; it starts at zero, and the quoter counts.
    pub counts: GlftCounts,
}

impl GlftQuoter {
    /// A quoter of the exact quotes, as [`GlftForm::Exact`], with no counts yet.
    #[must_use]
    pub fn new(
        config: GLFTConfig,
        volatility: Decimal,
        size: Decimal,
        session_start_ms: u64,
    ) -> Self {
        Self {
            config,
            volatility,
            size,
            session_start_ms,
            form: GlftForm::Exact,
            counts: GlftCounts::default(),
        }
    }

    /// Sets the form.
    #[must_use]
    pub fn with_form(mut self, form: GlftForm) -> Self {
        self.form = form;
        self
    }

    fn exact(
        &mut self,
        mid: Decimal,
        inventory: Decimal,
        elapsed: u64,
    ) -> MMResult<Result<GlftQuotes, MMError>> {
        match GLFTStrategy::exact_quotes(mid, inventory, &self.config, self.volatility, elapsed)? {
            ExactQuotes::Quotes(quotes) => Ok(Ok(quotes)),
            ExactQuotes::OverStepCap {
                steps_needed,
                budget,
            } => {
                self.counts.over_step_cap = self.counts.over_step_cap.saturating_add(1);
                Ok(Err(over_cap_error(steps_needed, budget)))
            }
        }
    }

    fn closed_form(
        &mut self,
        mid: Decimal,
        inventory: Decimal,
    ) -> MMResult<(GlftQuotes, GlftQuoteForm)> {
        let quotes =
            GLFTStrategy::closed_form_quotes(mid, inventory, &self.config, self.volatility)?;
        Ok((quotes, GlftQuoteForm::ClosedForm))
    }

    /// The quotes in the stated form, counted.
    fn quotes(&mut self, mid: Decimal, inventory: Decimal, elapsed: u64) -> MMResult<GlftQuotes> {
        self.counts.last = None;
        let (quotes, form) = match self.form {
            GlftForm::ClosedForm => self.closed_form(mid, inventory)?,
            GlftForm::Exact | GlftForm::ExactOrClosedForm => {
                match self.exact(mid, inventory, elapsed)? {
                    Ok(quotes) => (quotes, GlftQuoteForm::Exact),
                    Err(_) if self.form == GlftForm::ExactOrClosedForm => {
                        self.closed_form(mid, inventory)?
                    }
                    // The error the exact quote has always given there.
                    Err(over_cap) => return Err(over_cap),
                }
            }
        };
        match form {
            GlftQuoteForm::Exact => self.counts.exact = self.counts.exact.saturating_add(1),
            GlftQuoteForm::ClosedForm => {
                self.counts.closed_form = self.counts.closed_form.saturating_add(1);
            }
        }
        self.counts.last = Some(form);
        Ok(quotes)
    }
}

impl QuoteSource for GlftQuoter {
    fn quote(
        &mut self,
        obs: &Observation,
        mid: Decimal,
        inventory: Decimal,
    ) -> MMResult<SourceQuote> {
        let elapsed = elapsed_ms(obs, self.session_start_ms)?;
        let quotes = self.quotes(mid, inventory, elapsed)?;
        match (quotes.bid, quotes.ask) {
            (Some(bid), Some(ask)) => Ok(SourceQuote::Both(Quote {
                bid_price: bid,
                bid_size: self.size,
                ask_price: ask,
                ask_size: self.size,
                timestamp: obs.ts_ns / 1_000_000,
            })),
            (Some(bid), None) => Ok(SourceQuote::Bid(bid, self.size)),
            (None, Some(ask)) => Ok(SourceQuote::Ask(ask, self.size)),
            (None, None) => Err(MMError::InvalidQuoteGeneration(format!(
                "glft quote with no side at inventory {inventory}"
            ))),
        }
    }

    fn validate(&self, _tick: Decimal, lot: Decimal) -> MMResult<()> {
        if self.config.lot_size != lot {
            return Err(MMError::InvalidConfiguration(format!(
                "glft lot_size {} differs from the kernel lot {lot}",
                self.config.lot_size
            )));
        }
        if self.size != self.config.lot_size {
            return Err(MMError::InvalidConfiguration(format!(
                "glft order size {} must be one model lot ({})",
                self.size, self.config.lot_size
            )));
        }
        Ok(())
    }
}

/// Drives a [`QuoteSource`] from the observations of one kernel.
#[derive(Debug, Clone)]
pub struct QuoteAdapter<S> {
    source: S,
    tick: Unit,
    lot: Unit,
    max_price: u64,
    max_lots: u64,
    default_order_size: Decimal,
    kernel_fp: u128,
}

impl<S: QuoteSource> QuoteAdapter<S> {
    /// An adapter bound to `kernel`: the tick, lot and bounds come from it.
    /// `default_order_size` replaces a size that is not positive, as the
    /// engine's `BacktestConfig::default_order_size`.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] if the strategy does not
    /// match the kernel grid ([`QuoteSource::validate`]).
    pub fn new<const ORD: usize, const MSG: usize, const DLV: usize, const OBS: usize>(
        source: S,
        kernel: &FinancialKernel<'_, ORD, MSG, DLV, OBS>,
        default_order_size: Decimal,
    ) -> MMResult<Self> {
        let cfg = &kernel.cfg;
        source.validate(cfg.tick.value, cfg.lot.value)?;
        Ok(Self {
            source,
            tick: cfg.tick,
            lot: cfg.lot,
            max_price: cfg.p,
            max_lots: cfg.q,
            default_order_size,
            kernel_fp: cfg.fingerprint,
        })
    }

    /// The wrapped strategy.
    #[must_use]
    pub fn source(&self) -> &S {
        &self.source
    }

    /// The action for the observation of `kernel`; a kernel other than the
    /// one the adapter was built for gives `QuoteError(ERR_KERNEL_MISMATCH)`.
    pub fn act<const ORD: usize, const MSG: usize, const DLV: usize, const OBS: usize>(
        &mut self,
        kernel: &FinancialKernel<'_, ORD, MSG, DLV, OBS>,
        obs: &Observation,
    ) -> KernelAction {
        if kernel.config_fingerprint() != self.kernel_fp {
            return KernelAction::QuoteError(ERR_KERNEL_MISMATCH);
        }
        let mid = self
            .tick
            .to_decimal(i128::from(obs.bid))
            .zip(self.tick.to_decimal(i128::from(obs.ask)))
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

    /// Converts a strategy quote as the engine's `replace_quotes` does.
    #[must_use]
    pub fn to_action(&self, quote: &SourceQuote) -> KernelAction {
        match quote {
            SourceQuote::Both(quote) => self.two_sided(quote),
            SourceQuote::Bid(price, size) => self.one_sided(QuoteSide::Bid, *price, *size),
            SourceQuote::Ask(price, size) => self.one_sided(QuoteSide::Ask, *price, *size),
        }
    }

    fn two_sided(&self, quote: &Quote) -> KernelAction {
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

    fn one_sided(&self, side: QuoteSide, price: Decimal, size: Decimal) -> KernelAction {
        let up = match side {
            QuoteSide::Bid => false,
            QuoteSide::Ask => true,
        };
        let Some(price) = self.ticks(round_to_step(price, self.tick.value, up)) else {
            return KernelAction::QuoteError(ERR_PRICE_OUT_OF_RANGE);
        };
        let Some(lots) = self.lots(size) else {
            return KernelAction::QuoteError(ERR_SIZE_OUT_OF_RANGE);
        };
        KernelAction::OneSided { side, price, lots }
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
