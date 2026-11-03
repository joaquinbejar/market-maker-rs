//! Cleaning of option quotes before calibration.
//!
//! Every quote gets a [`QuoteVerdict`] with its reason; nothing is dropped
//! silently and nothing rejected enters the fit. Times are milliseconds since
//! the Unix epoch, UTC; prices are `Decimal` per unit of underlying.
//!
//! In order:
//!
//! 1. **Causality** (global): a quote, or its underlying, observed or
//!    received after the cutoff is an error for the whole set, and no
//!    surface may be built from it.
//! 2. **Per quote**: [`Rejection::Invalid`] (negative bid, non-positive ask,
//!    strike or underlying), [`Rejection::Crossed`] (`bid > ask`),
//!    [`Rejection::Locked`] (`bid == ask`: no band, so no weight),
//!    [`Rejection::Expired`] (expiry not after the cutoff),
//!    [`Rejection::Stale`] (`cutoff - observed_at > max_age_ms`),
//!    [`Rejection::SpotLag`] (quote and underlying observed more than
//!    `max_spot_lag_ms` apart) and [`Rejection::TooWide`] (`(ask - bid) /
//!    mid` above the maximum).
//! 3. **Duplicates** by `(strike, expiry, style)`: the latest observation
//!    wins and the others are [`Rejection::Superseded`]; two latest
//!    observations with different prices are both [`Rejection::Conflicting`].
//! 4. **Put-call parity**, for a call and a put at the same strike and expiry
//!    that share the same underlying observation:
//!    `|C_mid - P_mid - DF (F - K)| > (spread_C + spread_P) / 2 + tolerance`
//!    rejects both as [`Rejection::ParityInconsistent`]. A pair on different
//!    underlying observations is not compared and both are flagged
//!    [`QuoteFlag::ParityNotComparable`].
//! 5. **Side**: the out of the money side enters the fit (puts for `k < 0`,
//!    calls for `k >= 0`) and the other side is kept for parity only. When
//!    only the in the money side survives, it is used and flagged
//!    [`QuoteFlag::UsedItmSide`]: intrinsic value dominates its price, so its
//!    implied volatility is less precise.
//!
//! With `tau = (expiry - cutoff) / YEAR_MS`, `F = S e^{(r - q) tau}`,
//! `DF = e^{-r tau}` and `k = ln(K / F)`, in `f64`, every value checked
//! finite. Price bounds and implied volatility come after cleaning.

use optionstratlib::OptionStyle;
use rust_decimal::prelude::ToPrimitive;

use super::store::tau_between;
use crate::Decimal;
use crate::types::error::{MMError, MMResult};

/// One quote of one option, with the underlying price it was quoted
/// against.
#[derive(Debug, Clone, PartialEq)]
pub struct OptionQuote {
    /// Strike.
    pub strike: Decimal,
    /// Expiry, ms.
    pub expiry_at: u64,
    /// Call or put.
    pub style: OptionStyle,
    /// Best bid (0 when there is none).
    pub bid: Decimal,
    /// Best ask.
    pub ask: Decimal,
    /// When the quote was observed at the venue, ms.
    pub observed_at: u64,
    /// When the quote was received, ms.
    pub received_at: u64,
    /// Underlying price.
    pub underlying: Decimal,
    /// When the underlying price was observed, ms.
    pub underlying_observed_at: u64,
    /// When the underlying price was received, ms.
    pub underlying_received_at: u64,
}

/// Rates for the forward and the discount, continuously compounded.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rates {
    /// Risk free rate `r`.
    pub rate: Decimal,
    /// Carry or dividend yield `q`.
    pub carry: Decimal,
}

/// Thresholds of the cleaning.
#[derive(Debug, Clone, PartialEq)]
pub struct CleaningConfig {
    /// Largest `cutoff - observed_at`, ms.
    pub max_age_ms: u64,
    /// Largest distance between a quote's and its underlying's observation,
    /// ms.
    pub max_spot_lag_ms: u64,
    /// Largest `(ask - bid) / mid`.
    pub max_relative_spread: Decimal,
    /// Absolute tolerance added to the parity bound, in price units.
    pub parity_tolerance: Decimal,
}

/// Why a quote was rejected.
#[derive(Debug, Clone, PartialEq)]
pub enum Rejection {
    /// A value outside its domain.
    Invalid(String),
    /// `bid > ask`.
    Crossed,
    /// `bid == ask`: a quote with no band, whose weight
    /// `1 / band^2` would dwarf every other quote of its slice.
    Locked,
    /// The expiry is not after the cutoff.
    Expired,
    /// Observed `age_ms` before the cutoff, above the maximum.
    Stale {
        /// `cutoff - observed_at`.
        age_ms: u64,
    },
    /// Quote and underlying observed `lag_ms` apart, above the maximum.
    SpotLag {
        /// `|observed_at - underlying_observed_at|`.
        lag_ms: u64,
    },
    /// `(ask - bid) / mid` above the maximum.
    TooWide {
        /// The relative spread.
        relative: Decimal,
    },
    /// A later observation of the same option exists.
    Superseded,
    /// Another observation of the same option at the same time has other
    /// prices.
    Conflicting,
    /// The call and put at this strike break put-call parity.
    ParityInconsistent {
        /// `|C_mid - P_mid - DF (F - K)|`.
        gap: f64,
        /// `(spread_C + spread_P) / 2 + tolerance`.
        allowed: f64,
    },
}

/// A condition worth knowing that does not reject the quote.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuoteFlag {
    /// The in the money side is used because the other side is missing.
    UsedItmSide,
    /// Its pair was on another underlying observation; parity not checked.
    ParityNotComparable,
}

/// The outcome for one input quote.
#[derive(Debug, Clone, PartialEq)]
pub struct QuoteVerdict {
    /// Index of the quote in the input.
    pub index: usize,
    /// Why it was rejected, if it was.
    pub rejection: Option<Rejection>,
    /// Flags.
    pub flags: Vec<QuoteFlag>,
    /// Whether it enters the fit.
    pub used_in_fit: bool,
}

/// A quote that enters the fit, with its derived values.
#[derive(Debug, Clone, PartialEq)]
pub struct CleanQuote {
    /// Index of the quote in the input.
    pub index: usize,
    /// The quote.
    pub quote: OptionQuote,
    /// Years from the cutoff to expiry.
    pub tau: f64,
    /// Forward `S e^{(r - q) tau}`.
    pub forward: f64,
    /// Discount factor `e^{-r tau}`.
    pub discount: f64,
    /// `ln(K / F)`.
    pub k: f64,
}

/// The cleaning of one quote set.
#[derive(Debug, Clone, PartialEq)]
pub struct CleaningReport {
    /// The cutoff, ms.
    pub cutoff: u64,
    /// One verdict per input quote, in input order.
    pub verdicts: Vec<QuoteVerdict>,
    /// The quotes that enter the fit, by expiry then `k`.
    pub fit_set: Vec<CleanQuote>,
}

fn to_f64(name: &str, v: Decimal) -> MMResult<f64> {
    v.to_f64()
        .filter(|x| x.is_finite())
        .ok_or_else(|| MMError::NumericalError(format!("{name} {v} has no finite f64 value")))
}

fn finite(name: &str, v: f64) -> MMResult<f64> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err(MMError::NumericalError(format!(
            "{name} is not finite: {v}"
        )))
    }
}

/// Mid and spread, checked.
fn mid_spread(q: &OptionQuote) -> MMResult<(Decimal, Decimal)> {
    let overflow = || MMError::NumericalError("quote mid or spread overflowed".to_string());
    let sum = q.bid.checked_add(q.ask).ok_or_else(overflow)?;
    let mid = sum.checked_div(Decimal::TWO).ok_or_else(overflow)?;
    let spread = q.ask.checked_sub(q.bid).ok_or_else(overflow)?;
    Ok((mid, spread))
}

fn per_quote(q: &OptionQuote, cutoff: u64, config: &CleaningConfig) -> MMResult<Option<Rejection>> {
    if q.bid.is_sign_negative() && !q.bid.is_zero() {
        return Ok(Some(Rejection::Invalid(format!("negative bid {}", q.bid))));
    }
    for (name, v) in [
        ("ask", q.ask),
        ("strike", q.strike),
        ("underlying", q.underlying),
    ] {
        if v <= Decimal::ZERO {
            return Ok(Some(Rejection::Invalid(format!("non-positive {name} {v}"))));
        }
    }
    if q.bid > q.ask {
        return Ok(Some(Rejection::Crossed));
    }
    if q.bid == q.ask {
        return Ok(Some(Rejection::Locked));
    }
    if q.expiry_at <= cutoff {
        return Ok(Some(Rejection::Expired));
    }
    let age_ms = cutoff.saturating_sub(q.observed_at);
    if age_ms > config.max_age_ms {
        return Ok(Some(Rejection::Stale { age_ms }));
    }
    let lag_ms = q.observed_at.abs_diff(q.underlying_observed_at);
    if lag_ms > config.max_spot_lag_ms {
        return Ok(Some(Rejection::SpotLag { lag_ms }));
    }
    let (mid, spread) = mid_spread(q)?;
    let relative = spread
        .checked_div(mid)
        .ok_or_else(|| MMError::NumericalError("relative spread overflowed".to_string()))?;
    if relative > config.max_relative_spread {
        return Ok(Some(Rejection::TooWide { relative }));
    }
    Ok(None)
}

/// Cleans `quotes` at `cutoff`; see the [module docs](self).
///
/// # Errors
///
/// - [`MMError::InvalidTimestamp`] when any quote or underlying was
///   observed or received after `cutoff`: the whole set is refused.
/// - [`MMError::InvalidConfiguration`] for negative thresholds.
/// - [`MMError::NumericalError`] when a derived value has no finite `f64`.
pub fn clean_quotes(
    quotes: &[OptionQuote],
    cutoff: u64,
    rates: Rates,
    config: &CleaningConfig,
) -> MMResult<CleaningReport> {
    if config.max_relative_spread.is_sign_negative() || config.parity_tolerance.is_sign_negative() {
        return Err(MMError::InvalidConfiguration(
            "cleaning thresholds must be non-negative".to_string(),
        ));
    }
    for (i, q) in quotes.iter().enumerate() {
        let latest = q
            .observed_at
            .max(q.received_at)
            .max(q.underlying_observed_at)
            .max(q.underlying_received_at);
        if latest > cutoff {
            return Err(MMError::InvalidTimestamp(format!(
                "quote {i} carries data from {latest}, after the cutoff {cutoff}"
            )));
        }
    }
    let r = to_f64("rate", rates.rate)?;
    let q_carry = to_f64("carry", rates.carry)?;

    let mut verdicts: Vec<QuoteVerdict> = Vec::with_capacity(quotes.len());
    for (index, q) in quotes.iter().enumerate() {
        verdicts.push(QuoteVerdict {
            index,
            rejection: per_quote(q, cutoff, config)?,
            flags: Vec::new(),
            used_in_fit: false,
        });
    }

    // Duplicates among the survivors.
    let alive = |v: &QuoteVerdict| v.rejection.is_none();
    let same_option = |a: &OptionQuote, b: &OptionQuote| {
        a.strike == b.strike && a.expiry_at == b.expiry_at && a.style == b.style
    };
    let mut decisions: Vec<(usize, Rejection)> = Vec::new();
    for (i, qi) in quotes.iter().enumerate() {
        if !verdicts.get(i).is_some_and(alive) {
            continue;
        }
        for (j, qj) in quotes.iter().enumerate() {
            if i == j || !verdicts.get(j).is_some_and(alive) || !same_option(qi, qj) {
                continue;
            }
            if qj.observed_at > qi.observed_at {
                decisions.push((i, Rejection::Superseded));
                break;
            }
            if qj.observed_at == qi.observed_at {
                if qj.bid != qi.bid || qj.ask != qi.ask {
                    decisions.push((i, Rejection::Conflicting));
                    break;
                }
                if j < i {
                    // The same observation twice: the first one stands.
                    decisions.push((i, Rejection::Superseded));
                    break;
                }
            }
        }
    }
    for (i, rejection) in decisions {
        if let Some(v) = verdicts.get_mut(i) {
            v.rejection = Some(rejection);
        }
    }

    // Derived values of the survivors.
    let mut derived: Vec<Option<CleanQuote>> = vec![None; quotes.len()];
    for (index, q) in quotes.iter().enumerate() {
        if !verdicts.get(index).is_some_and(alive) {
            continue;
        }
        // Alive quotes expire after the cutoff (others are Expired).
        let tau = tau_between(cutoff, q.expiry_at)?;
        let spot = to_f64("underlying", q.underlying)?;
        let strike = to_f64("strike", q.strike)?;
        let forward = finite("forward", spot * ((r - q_carry) * tau).exp())?;
        let discount = finite("discount", (-r * tau).exp())?;
        let k = finite("k", (strike / forward).ln())?;
        if let Some(slot) = derived.get_mut(index) {
            *slot = Some(CleanQuote {
                index,
                quote: q.clone(),
                tau,
                forward,
                discount,
                k,
            });
        }
    }

    // Parity and side selection per (strike, expiry).
    let tolerance = to_f64("parity tolerance", config.parity_tolerance)?;
    let mut parity_rejects: Vec<(usize, Rejection)> = Vec::new();
    let mut flags: Vec<(usize, QuoteFlag)> = Vec::new();
    let mut fit_set: Vec<CleanQuote> = Vec::new();
    for (i, slot) in derived.iter().enumerate() {
        let Some(call) = slot.as_ref().filter(|c| c.quote.style == OptionStyle::Call) else {
            continue;
        };
        let put = derived.iter().flatten().find(|p| {
            p.quote.style == OptionStyle::Put
                && p.quote.strike == call.quote.strike
                && p.quote.expiry_at == call.quote.expiry_at
        });
        let Some(put) = put else { continue };
        if call.quote.underlying_observed_at != put.quote.underlying_observed_at
            || call.quote.underlying != put.quote.underlying
        {
            flags.push((i, QuoteFlag::ParityNotComparable));
            flags.push((put.index, QuoteFlag::ParityNotComparable));
            continue;
        }
        let (c_mid, c_spread) = mid_spread(&call.quote)?;
        let (p_mid, p_spread) = mid_spread(&put.quote)?;
        let strike = to_f64("strike", call.quote.strike)?;
        let gap = finite(
            "parity gap",
            (to_f64("call mid", c_mid)?
                - to_f64("put mid", p_mid)?
                - call.discount * (call.forward - strike))
                .abs(),
        )?;
        let allowed =
            (to_f64("call spread", c_spread)? + to_f64("put spread", p_spread)?) / 2.0 + tolerance;
        if gap > allowed {
            parity_rejects.push((i, Rejection::ParityInconsistent { gap, allowed }));
            parity_rejects.push((put.index, Rejection::ParityInconsistent { gap, allowed }));
        }
    }
    for (i, rejection) in parity_rejects {
        if let Some(v) = verdicts.get_mut(i) {
            v.rejection = Some(rejection);
        }
    }
    for c in derived.iter().flatten() {
        if !verdicts.get(c.index).is_some_and(alive) {
            continue;
        }
        let otm_style = if c.k < 0.0 {
            OptionStyle::Put
        } else {
            OptionStyle::Call
        };
        if c.quote.style == otm_style {
            fit_set.push(c.clone());
            continue;
        }
        let otm_alive = derived.iter().flatten().any(|o| {
            o.quote.style == otm_style
                && o.quote.strike == c.quote.strike
                && o.quote.expiry_at == c.quote.expiry_at
                && verdicts.get(o.index).is_some_and(alive)
        });
        if !otm_alive {
            flags.push((c.index, QuoteFlag::UsedItmSide));
            fit_set.push(c.clone());
        }
    }
    for (i, flag) in flags {
        if let Some(v) = verdicts.get_mut(i)
            && !v.flags.contains(&flag)
        {
            v.flags.push(flag);
        }
    }
    for c in &fit_set {
        if let Some(v) = verdicts.get_mut(c.index) {
            v.used_in_fit = true;
        }
    }
    fit_set.sort_by(|a, b| {
        a.quote
            .expiry_at
            .cmp(&b.quote.expiry_at)
            .then(a.k.total_cmp(&b.k))
    });
    Ok(CleaningReport {
        cutoff,
        verdicts,
        fit_set,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    const DAY: u64 = 86_400_000;
    const CUTOFF: u64 = 1_000 * DAY;

    fn quote(strike: Decimal, style: OptionStyle, bid: Decimal, ask: Decimal) -> OptionQuote {
        OptionQuote {
            strike,
            expiry_at: CUTOFF + 30 * DAY,
            style,
            bid,
            ask,
            observed_at: CUTOFF - 1000,
            received_at: CUTOFF - 900,
            underlying: dec!(100),
            underlying_observed_at: CUTOFF - 1000,
            underlying_received_at: CUTOFF - 900,
        }
    }

    fn config() -> CleaningConfig {
        CleaningConfig {
            max_age_ms: 60_000,
            max_spot_lag_ms: 500,
            max_relative_spread: dec!(0.5),
            parity_tolerance: dec!(0.01),
        }
    }

    fn rates() -> Rates {
        Rates {
            rate: Decimal::ZERO,
            carry: Decimal::ZERO,
        }
    }

    fn rejection(report: &CleaningReport, i: usize) -> Option<&Rejection> {
        report.verdicts.get(i).and_then(|v| v.rejection.as_ref())
    }

    #[test]
    fn test_clean_quotes_future_data_is_a_global_error() {
        let mut late = quote(dec!(100), OptionStyle::Call, dec!(2), dec!(2.2));
        late.received_at = CUTOFF + 1;
        let ok = quote(dec!(110), OptionStyle::Call, dec!(1), dec!(1.1));
        assert!(matches!(
            clean_quotes(&[ok.clone(), late], CUTOFF, rates(), &config()),
            Err(MMError::InvalidTimestamp(_))
        ));
        let mut spot_late = ok.clone();
        spot_late.underlying_observed_at = CUTOFF + 1;
        assert!(clean_quotes(&[spot_late], CUTOFF, rates(), &config()).is_err());
        let mut spot_received_late = ok.clone();
        spot_received_late.underlying_received_at = CUTOFF + 1;
        assert!(clean_quotes(&[spot_received_late], CUTOFF, rates(), &config()).is_err());
        // A future copy of a quote is not hidden by de-duplication: the
        // causality check runs first, on every quote.
        let mut future_copy = ok.clone();
        future_copy.observed_at = CUTOFF + 5;
        assert!(matches!(
            clean_quotes(&[ok, future_copy], CUTOFF, rates(), &config()),
            Err(MMError::InvalidTimestamp(_))
        ));
    }

    #[test]
    fn test_clean_quotes_per_quote_rejections() {
        let mut stale = quote(dec!(120), OptionStyle::Call, dec!(0.5), dec!(0.6));
        stale.observed_at = CUTOFF - 120_000;
        stale.underlying_observed_at = stale.observed_at;
        let mut lagged = quote(dec!(125), OptionStyle::Call, dec!(0.4), dec!(0.5));
        lagged.underlying_observed_at = lagged.observed_at - 1_000;
        let mut expired = quote(dec!(130), OptionStyle::Call, dec!(0.3), dec!(0.4));
        expired.expiry_at = CUTOFF;
        let quotes = vec![
            quote(dec!(105), OptionStyle::Call, dec!(2), dec!(1.9)),
            quote(dec!(110), OptionStyle::Call, dec!(-0.1), dec!(1)),
            quote(dec!(115), OptionStyle::Call, dec!(0.1), dec!(1)),
            stale,
            lagged,
            expired,
            quote(dec!(135), OptionStyle::Call, dec!(0.2), dec!(0.2)),
        ];
        let report = clean_quotes(&quotes, CUTOFF, rates(), &config()).unwrap();
        assert_eq!(rejection(&report, 0), Some(&Rejection::Crossed));
        assert!(matches!(rejection(&report, 1), Some(Rejection::Invalid(_))));
        assert!(matches!(
            rejection(&report, 2),
            Some(Rejection::TooWide { .. })
        ));
        assert_eq!(
            rejection(&report, 3),
            Some(&Rejection::Stale { age_ms: 120_000 })
        );
        assert_eq!(
            rejection(&report, 4),
            Some(&Rejection::SpotLag { lag_ms: 1_000 })
        );
        assert_eq!(rejection(&report, 5), Some(&Rejection::Expired));
        assert_eq!(rejection(&report, 6), Some(&Rejection::Locked));
        assert!(report.fit_set.is_empty());
        assert_eq!(report.verdicts.len(), 7);
    }

    #[test]
    fn test_clean_quotes_duplicates_superseded_or_conflicting() {
        let old = quote(dec!(110), OptionStyle::Call, dec!(1), dec!(1.1));
        let mut new = old.clone();
        new.observed_at += 100;
        new.underlying_observed_at += 100;
        new.bid = dec!(1.05);
        let mut twin = new.clone();
        twin.ask = dec!(1.2);
        let report = clean_quotes(&[old.clone(), new.clone()], CUTOFF, rates(), &config()).unwrap();
        assert_eq!(rejection(&report, 0), Some(&Rejection::Superseded));
        assert_eq!(rejection(&report, 1), None);
        let report = clean_quotes(&[old, new, twin], CUTOFF, rates(), &config()).unwrap();
        assert_eq!(rejection(&report, 1), Some(&Rejection::Conflicting));
        assert_eq!(rejection(&report, 2), Some(&Rejection::Conflicting));
    }

    #[test]
    fn test_clean_quotes_parity_breaks_reject_both_and_otm_side_enters_fit() {
        // r = q = 0, F = 100: C - P = 100 - K.
        let quotes = vec![
            quote(dec!(90), OptionStyle::Call, dec!(10.9), dec!(11.1)),
            quote(dec!(90), OptionStyle::Put, dec!(0.9), dec!(1.1)),
            quote(dec!(110), OptionStyle::Call, dec!(0.9), dec!(1.1)),
            // C - P = 1 - 14 = -13, parity says -10.
            quote(dec!(110), OptionStyle::Put, dec!(13.9), dec!(14.1)),
        ];
        let report = clean_quotes(&quotes, CUTOFF, rates(), &config()).unwrap();
        assert_eq!(rejection(&report, 0), None);
        assert_eq!(rejection(&report, 1), None);
        assert!(matches!(
            rejection(&report, 2),
            Some(Rejection::ParityInconsistent { .. })
        ));
        assert!(matches!(
            rejection(&report, 3),
            Some(Rejection::ParityInconsistent { .. })
        ));
        // Only the 90 put (out of the money) enters the fit.
        assert_eq!(report.fit_set.len(), 1);
        assert_eq!(report.fit_set.first().map(|c| c.index), Some(1));
        assert!(!report.verdicts.first().unwrap().used_in_fit);
    }

    #[test]
    fn test_clean_quotes_pairs_on_different_spots_are_not_compared() {
        let call = quote(dec!(110), OptionStyle::Call, dec!(0.9), dec!(1.1));
        let mut put = quote(dec!(110), OptionStyle::Put, dec!(13.9), dec!(14.1));
        put.underlying = dec!(100.5);
        let report = clean_quotes(&[call, put], CUTOFF, rates(), &config()).unwrap();
        assert_eq!(rejection(&report, 0), None);
        assert!(
            report
                .verdicts
                .iter()
                .all(|v| v.flags.contains(&QuoteFlag::ParityNotComparable))
        );
    }

    #[test]
    fn test_clean_quotes_itm_side_alone_is_used_and_flagged() {
        // k < 0 (strike 90 below F = 100) wants the put; only the call exists.
        let report = clean_quotes(
            &[quote(dec!(90), OptionStyle::Call, dec!(10.9), dec!(11.1))],
            CUTOFF,
            rates(),
            &config(),
        )
        .unwrap();
        assert_eq!(report.fit_set.len(), 1);
        assert_eq!(
            report.verdicts.first().unwrap().flags,
            vec![QuoteFlag::UsedItmSide]
        );
    }

    #[test]
    fn test_clean_quotes_derives_forward_and_k() {
        let r = Rates {
            rate: dec!(0.05),
            carry: dec!(0.01),
        };
        let report = clean_quotes(
            &[quote(dec!(110), OptionStyle::Call, dec!(0.9), dec!(1.1))],
            CUTOFF,
            r,
            &config(),
        )
        .unwrap();
        let c = report.fit_set.first().unwrap();
        let tau = 30.0 / 365.0;
        assert!((c.tau - tau).abs() < 1e-15);
        assert!((c.forward - 100.0 * (0.04 * tau).exp()).abs() < 1e-12);
        assert!((c.discount - (-0.05 * tau).exp()).abs() < 1e-15);
        assert!((c.k - (110.0 / c.forward).ln()).abs() < 1e-15);
    }
}
