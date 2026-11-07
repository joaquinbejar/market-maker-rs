//! `GlftQuoter`'s forms (#264): over the matrix exponential's step cap the exact quote is
//! refused, typed and counted, and the caller states whether the closed form takes its
//! place. All data here is synthetic.

#![cfg(feature = "kernel")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use market_maker_rs::Decimal;
use market_maker_rs::backtest::MarketTick;
use market_maker_rs::backtest::kernel::{
    FinancialKernel, GlftForm, GlftQuoteForm, GlftQuoter, KernelAction, KernelConfig,
    KernelDataset, QuoteAdapter, quote_error_code,
};
use market_maker_rs::dec;
use market_maker_rs::strategy::glft::{ExactQuotes, GLFTConfig, GLFTStrategy};
use market_maker_rs::types::error::MMError;

const DAY_MS: u64 = 86_400_000;

/// The setting of #264: a one-day horizon, risk aversion 0.001, `k` 50, `A` 1.5, `Q` 5.
fn config() -> GLFTConfig {
    GLFTConfig::new(
        dec!(0.001),
        dec!(50),
        dec!(1.5),
        5,
        dec!(1),
        DAY_MS,
        dec!(0),
    )
    .unwrap()
}

const VOLATILITY: Decimal = dec!(50);

fn over_cap(elapsed_ms: u64) -> bool {
    matches!(
        GLFTStrategy::exact_quotes(dec!(100), dec!(0), &config(), VOLATILITY, elapsed_ms).unwrap(),
        ExactQuotes::OverStepCap { .. }
    )
}

/// The first whole second from which the rest of the day fits under the step cap.
fn boundary_ms() -> u64 {
    assert!(over_cap(0), "a whole day must be over the cap");
    assert!(!over_cap(DAY_MS - 1_000));
    let (mut lo, mut hi) = (0u64, (DAY_MS - 1_000) / 1_000);
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if over_cap(mid * 1_000) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    hi * 1_000
}

#[test]
fn test_the_exact_quote_tells_the_step_cap_apart_and_keeps_its_error() {
    // Over the cap: typed, and the old call's error is what it was.
    let outcome = GLFTStrategy::exact_quotes(dec!(100), dec!(0), &config(), VOLATILITY, 0).unwrap();
    let ExactQuotes::OverStepCap {
        steps_needed,
        budget,
    } = outcome
    else {
        panic!("{outcome:?}");
    };
    assert!(steps_needed > f64::from(budget));
    let error =
        GLFTStrategy::calculate_optimal_quotes(dec!(100), dec!(0), &config(), VOLATILITY, 0)
            .unwrap_err();
    assert!(
        matches!(&error, MMError::NumericalError(m) if m.contains("horizon too stiff")),
        "{error:?}"
    );
    // Under it: the same quotes as the old call.
    let late = DAY_MS - 60_000;
    let ExactQuotes::Quotes(quotes) =
        GLFTStrategy::exact_quotes(dec!(100), dec!(2), &config(), VOLATILITY, late).unwrap()
    else {
        panic!("under the cap");
    };
    assert_eq!(
        quotes,
        GLFTStrategy::calculate_optimal_quotes(dec!(100), dec!(2), &config(), VOLATILITY, late)
            .unwrap()
    );
}

/// Rows from five seconds before the boundary to five after, one a second, touching so
/// that quotes fill and the inventory moves.
fn data(cfg: &KernelConfig, boundary: u64) -> KernelDataset {
    let ticks: Vec<MarketTick> = (0..10_u64)
        .map(|i| {
            let mid = if i % 2 == 0 { dec!(100) } else { dec!(100.02) };
            MarketTick::new(
                boundary - 5_000 + i * 1_000,
                mid - dec!(0.01),
                dec!(5),
                mid + dec!(0.01),
                dec!(5),
            )
        })
        .collect();
    KernelDataset::from_ticks(&ticks, cfg).unwrap()
}

struct Run {
    actions: Vec<KernelAction>,
    forms: Vec<Option<GlftQuoteForm>>,
    quoter: GlftQuoter,
}

/// Runs a quoter of `form` over the rows around the boundary, and checks every quote it
/// gives against a fresh quoter of the form it says it used, at the same observation.
fn run(form: GlftForm) -> Run {
    let boundary = boundary_ms();
    let cfg = KernelConfig::new(dec!(0.01), dec!(1), dec!(1000), dec!(5), 100)
        .with_initial_capital(dec!(100000));
    let data = data(&cfg, boundary);
    let kernel: FinancialKernel<'_> = FinancialKernel::new(&cfg, &data).unwrap();
    let quoter = GlftQuoter::new(config(), VOLATILITY, dec!(1), 0).with_form(form);
    let mut adapter = QuoteAdapter::new(quoter, &kernel, dec!(1)).unwrap();
    let mut s = kernel.reset(1);
    let mut actions = Vec::new();
    let mut forms = Vec::new();
    while s.status().is_running() {
        let obs = kernel.observe(&s);
        let action = adapter.act(&kernel, &obs);
        let last = adapter.source().counts.last;
        if let Some(used) = last {
            let alone = match used {
                GlftQuoteForm::Exact => GlftForm::Exact,
                _ => GlftForm::ClosedForm,
            };
            let mut check = QuoteAdapter::new(
                GlftQuoter::new(config(), VOLATILITY, dec!(1), 0).with_form(alone),
                &kernel,
                dec!(1),
            )
            .unwrap();
            assert_eq!(check.act(&kernel, &obs), action);
        }
        actions.push(action);
        forms.push(last);
        kernel.step(&mut s, action);
    }
    Run {
        actions,
        forms,
        quoter: adapter.source().clone(),
    }
}

fn numerical() -> KernelAction {
    KernelAction::QuoteError(quote_error_code(&MMError::NumericalError(String::new())))
}

#[test]
fn test_the_exact_form_holds_over_the_cap_and_counts_it() {
    let r = run(GlftForm::Exact);
    let c = r.quoter.counts;
    assert!(c.over_step_cap > 0 && c.exact > 0);
    assert_eq!(c.closed_form, 0);
    assert_eq!(c.over_step_cap + c.exact, r.actions.len() as u64);
    let refused = r.actions.iter().filter(|a| **a == numerical()).count() as u64;
    assert_eq!(refused, c.over_step_cap);
    // The refusals are the decisions before the boundary, and they say so.
    for (action, form) in r.actions.iter().zip(&r.forms) {
        assert_eq!(*action == numerical(), form.is_none());
    }
}

#[test]
fn test_the_stated_fallback_quotes_every_decision_and_says_which_form() {
    let r = run(GlftForm::ExactOrClosedForm);
    let c = r.quoter.counts;
    assert!(c.over_step_cap > 0 && c.exact > 0);
    assert_eq!(c.closed_form, c.over_step_cap);
    assert_eq!(c.exact + c.closed_form, r.actions.len() as u64);
    assert!(
        r.actions
            .iter()
            .all(|a| !matches!(a, KernelAction::QuoteError(_)))
    );
    // The closed form comes first, then the exact one, at the boundary.
    let first_exact = r
        .forms
        .iter()
        .position(|f| *f == Some(GlftQuoteForm::Exact));
    let first_exact = first_exact.unwrap();
    assert!(
        r.forms[..first_exact]
            .iter()
            .all(|f| *f == Some(GlftQuoteForm::ClosedForm))
    );
    assert!(
        r.forms[first_exact..]
            .iter()
            .all(|f| *f == Some(GlftQuoteForm::Exact))
    );
}

#[test]
fn test_the_closed_form_quotes_every_decision_and_never_tries_the_exact_one() {
    let r = run(GlftForm::ClosedForm);
    let c = r.quoter.counts;
    assert_eq!((c.exact, c.over_step_cap), (0, 0));
    assert_eq!(c.closed_form, r.actions.len() as u64);
    assert!(
        r.forms
            .iter()
            .all(|f| *f == Some(GlftQuoteForm::ClosedForm))
    );
}

#[test]
fn test_a_new_quoter_is_the_exact_one_with_no_counts() {
    let q = GlftQuoter::new(config(), VOLATILITY, dec!(1), 0);
    assert_eq!(q.form, GlftForm::Exact);
    assert_eq!(q.counts, Default::default());
    assert_eq!(GlftForm::default(), GlftForm::Exact);
}

#[test]
fn test_a_refused_quote_after_a_given_one_leaves_no_form() {
    use market_maker_rs::backtest::kernel::QuoteSource;
    let boundary = boundary_ms();
    let cfg = KernelConfig::new(dec!(0.01), dec!(1), dec!(1000), dec!(5), 100);
    let data = data(&cfg, boundary);
    let kernel: FinancialKernel<'_> = FinancialKernel::new(&cfg, &data).unwrap();
    let s = kernel.reset(1);
    let mut late = kernel.observe(&s);
    late.ts_ns = (DAY_MS - 60_000) * 1_000_000;
    let mut early = late;
    early.ts_ns = 0;
    let mut q = GlftQuoter::new(config(), VOLATILITY, dec!(1), 0);
    assert!(q.quote(&late, dec!(100), dec!(0)).is_ok());
    assert_eq!(q.counts.last, Some(GlftQuoteForm::Exact));
    assert!(q.quote(&early, dec!(100), dec!(0)).is_err());
    assert_eq!(q.counts.last, None);
    assert_eq!((q.counts.exact, q.counts.over_step_cap), (1, 1));
}

/// `GlftQuoter` as it was before the forms: the exact quote, and its error as it was.
struct BeforeForms {
    config: GLFTConfig,
    volatility: Decimal,
    size: Decimal,
    session_start_ms: u64,
}

impl market_maker_rs::backtest::kernel::QuoteSource for BeforeForms {
    fn quote(
        &mut self,
        obs: &market_maker_rs::backtest::kernel::Observation,
        mid: Decimal,
        inventory: Decimal,
    ) -> market_maker_rs::types::error::MMResult<market_maker_rs::backtest::kernel::SourceQuote>
    {
        use market_maker_rs::backtest::kernel::SourceQuote;
        let now_ms = obs.ts_ns / 1_000_000;
        let elapsed = now_ms.checked_sub(self.session_start_ms).ok_or_else(|| {
            MMError::InvalidTimestamp(format!(
                "observation at {now_ms} ms precedes the session start {} ms",
                self.session_start_ms
            ))
        })?;
        let quotes = GLFTStrategy::calculate_optimal_quotes(
            mid,
            inventory,
            &self.config,
            self.volatility,
            elapsed,
        )?;
        match (quotes.bid, quotes.ask) {
            (Some(bid), Some(ask)) => {
                Ok(SourceQuote::Both(market_maker_rs::strategy::quote::Quote {
                    bid_price: bid,
                    bid_size: self.size,
                    ask_price: ask,
                    ask_size: self.size,
                    timestamp: obs.ts_ns / 1_000_000,
                }))
            }
            (Some(bid), None) => Ok(SourceQuote::Bid(bid, self.size)),
            (None, Some(ask)) => Ok(SourceQuote::Ask(ask, self.size)),
            (None, None) => Err(MMError::InvalidQuoteGeneration(format!(
                "glft quote with no side at inventory {inventory}"
            ))),
        }
    }
}

#[test]
fn test_the_default_form_gives_exactly_what_the_quoter_gave_before_the_forms() {
    use market_maker_rs::backtest::kernel::QuoteSource;
    // Over the cap, across it, and under it: rows from before the boundary to the end of the
    // day, so both the exact quotes and the exact errors are compared.
    let boundary = boundary_ms();
    let cfg = KernelConfig::new(dec!(0.01), dec!(1), dec!(1000), dec!(5), 1_000)
        .with_initial_capital(dec!(100000));
    let ticks: Vec<MarketTick> = (0..40_u64)
        .map(|i| {
            let ms = if i < 20 {
                boundary - 10_000 + i * 1_000
            } else {
                DAY_MS - 20_000 + (i - 20) * 1_000
            };
            let mid = dec!(100) + Decimal::from(i % 5) * dec!(0.01);
            MarketTick::new(ms, mid - dec!(0.01), dec!(5), mid + dec!(0.01), dec!(5))
        })
        .collect();
    let data = KernelDataset::from_ticks(&ticks, &cfg).unwrap();
    let kernel: FinancialKernel<'_> = FinancialKernel::new(&cfg, &data).unwrap();
    let mut now = QuoteAdapter::new(
        GlftQuoter::new(config(), VOLATILITY, dec!(1), 0),
        &kernel,
        dec!(1),
    )
    .unwrap();
    let before = || BeforeForms {
        config: config(),
        volatility: VOLATILITY,
        size: dec!(1),
        session_start_ms: 0,
    };
    let mut then = QuoteAdapter::new(before(), &kernel, dec!(1)).unwrap();
    let (mut direct_now, mut direct_then) =
        (GlftQuoter::new(config(), VOLATILITY, dec!(1), 0), before());
    let mut s = kernel.reset(3);
    let (mut errors, mut quotes) = (0, 0);
    while s.status().is_running() {
        let obs = kernel.observe(&s);
        let action = now.act(&kernel, &obs);
        assert_eq!(action, then.act(&kernel, &obs));
        // The quotes and the errors themselves, messages included.
        let mid = kernel.price(obs.bid + obs.ask).unwrap() / dec!(2);
        let a = direct_now.quote(&obs, mid, obs.position);
        assert_eq!(a, direct_then.quote(&obs, mid, obs.position));
        if a.is_ok() {
            quotes += 1;
        } else {
            errors += 1;
        }
        kernel.step(&mut s, action);
    }
    assert!(
        quotes > 0 && errors > 0,
        "{quotes} quotes and {errors} errors"
    );
}
