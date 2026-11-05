//! The lifecycle of a dated future: its expiry and its settlement, per contract.
//!
//! [`FutureTerms`] joins an [`InstrumentSpec`] with how the contract settles and
//! who publishes its settlement price. A specification alone does not say: both
//! are read from the contract's rules, and no convention common to all futures is
//! assumed. [`FutureCalendar`] orders contracts that were declared; it generates
//! no dates.
//!
//! # Phases
//!
//! | phase | when |
//! |---|---|
//! | [`FuturePhase::Trading`] | before the expiry |
//! | [`FuturePhase::AwaitingSettlement`] | at or after the expiry, not settled |
//! | [`FuturePhase::Settled`] | a ledger booked the settlement |
//!
//! The expiry itself is not trading, as a ledger refuses a fill at it. A clock
//! alone never reaches [`FuturePhase::Settled`]: only
//! `FutureTerms::phase_in_ledger` (feature `margin`) reports it, from the ledger's
//! own record. With
//! no settlement price a contract stays [`FuturePhase::AwaitingSettlement`] and no
//! position changes.
//!
//! # The settlement price
//!
//! A [`SettlementPrice`] is the only way into a settlement, and no function here
//! takes a mark or a last price, so nothing falls back on one. It cannot prove
//! where a price came from, though: the caller builds it, and
//! [`FutureTerms::check_settlement`] checks what it declares, namely the contract,
//! the source, the expiry it settles and that it was published at or after that
//! expiry. A last price labelled with the right source would pass.
//!
//! A settlement price is not on the order grid: Deribit delivered `BTC-4NOV26` at
//! 78044.4, off its 2.5 tick. It is never checked against the tick. In a ledger, its
//! digits must fit the instrument's explicit `price_scale`.
//!
//! How a venue computes the price is outside this module, which takes it as
//! published and does not reproduce or verify any averaging.
//!
//! # Cash settlement only
//!
//! [`SettlementMethod`] has `Cash`, the only method of a contract pinned so far.
//! Delivery waits for a pinned contract to check against and a ledger booking for
//! it. The outcome of a cash settlement is the realized PnL against the settlement
//! price, in the settlement asset, with no notional in it. A future receives no
//! funding: a ledger refuses a funding payment on one.

use financial_types::Side;
use rust_decimal::Decimal;

#[cfg(feature = "margin")]
use margin_rs::MarginError;
#[cfg(feature = "margin")]
use margin_rs::ledger::{ExpiryCashflow, ExpirySettlement, Ledger, PostingBooking};

use super::{ContractKind, InstrumentError, InstrumentId, InstrumentSpec, MoneyRatio};

/// How a future settles at expiry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SettlementMethod {
    /// The position closes at the settlement price, and its PnL is paid in the
    /// settlement asset.
    Cash,
}

/// Who publishes a contract's settlement price: a label, compared exactly.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SettlementSource(String);

impl SettlementSource {
    /// A source from its name, such as an API endpoint.
    ///
    /// # Errors
    ///
    /// [`FutureError::EmptySource`] if the name is empty or only whitespace.
    pub fn new(name: impl Into<String>) -> Result<Self, FutureError> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(FutureError::EmptySource);
        }
        Ok(Self(name))
    }

    /// The name of the source.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// A settlement price as published, with what identifies it.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SettlementPrice {
    instrument: InstrumentId,
    source: SettlementSource,
    expiry_ms: u64,
    price: Decimal,
    published_ms: u64,
}

impl SettlementPrice {
    /// A settlement price of `instrument`, for its expiry at `expiry_ms`, published
    /// by `source` at `published_ms`.
    ///
    /// # Errors
    ///
    /// [`FutureError::NotPositivePrice`] if the price is not positive.
    pub fn new(
        instrument: InstrumentId,
        source: SettlementSource,
        expiry_ms: u64,
        price: Decimal,
        published_ms: u64,
    ) -> Result<Self, FutureError> {
        if price <= Decimal::ZERO {
            return Err(FutureError::NotPositivePrice { price });
        }
        Ok(Self {
            instrument,
            source,
            expiry_ms,
            price,
            published_ms,
        })
    }

    /// The contract it settles.
    #[must_use]
    pub fn instrument(&self) -> &InstrumentId {
        &self.instrument
    }

    /// Who published it.
    #[must_use]
    pub fn source(&self) -> &SettlementSource {
        &self.source
    }

    /// The expiry it settles, in milliseconds since the epoch.
    #[must_use]
    pub fn expiry_ms(&self) -> u64 {
        self.expiry_ms
    }

    /// The price.
    #[must_use]
    pub fn price(&self) -> Decimal {
        self.price
    }

    /// When it was published, in milliseconds since the epoch.
    #[must_use]
    pub fn published_ms(&self) -> u64 {
        self.published_ms
    }
}

/// Where a future is in its lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum FuturePhase {
    /// Before the expiry.
    Trading,
    /// At or after the expiry, with no settlement booked.
    AwaitingSettlement,
    /// A ledger booked the settlement.
    Settled,
}

/// The outcome of a cash settlement of one position.
#[derive(Debug, Clone)]
pub struct CashSettlement {
    /// The realized PnL against the settlement price, exactly, in the settlement
    /// asset: positive is received.
    pub pnl: MoneyRatio,
    /// The settlement price.
    pub price: Decimal,
    /// The expiry settled: the date the position closes.
    pub expiry_ms: u64,
    /// When the price was published, which is the date a ledger books the cash.
    pub published_ms: u64,
}

/// Why a future or its settlement was refused.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum FutureError {
    /// The contract is not a dated future.
    #[error("{instrument} is a {kind:?}, not a dated future")]
    NotAFuture {
        /// The contract.
        instrument: String,
        /// What it is.
        kind: ContractKind,
    },
    /// A source without a name.
    #[error("a settlement source needs a name")]
    EmptySource,
    /// A settlement price that is not positive.
    #[error("settlement price {price} is not positive")]
    NotPositivePrice {
        /// The price.
        price: Decimal,
    },
    /// A settlement price of another contract.
    #[error("settlement price of {given} used for {instrument}")]
    WrongInstrument {
        /// The contract settled.
        instrument: String,
        /// The contract of the price.
        given: String,
    },
    /// A settlement price from another source.
    #[error("settlement price of {instrument} from {given}, and its source is {expected}")]
    WrongSource {
        /// The contract settled.
        instrument: String,
        /// The contract's source.
        expected: String,
        /// The source of the price.
        given: String,
    },
    /// A settlement price of another expiry: published later, it may still be
    /// another day's.
    #[error("settlement price of {instrument} for expiry {given}, and it expires at {expected}")]
    WrongExpiry {
        /// The contract settled.
        instrument: String,
        /// The contract's expiry.
        expected: u64,
        /// The expiry of the price.
        given: u64,
    },
    /// A settlement price published before the expiry it settles.
    #[error("settlement price of {instrument} published at {published_ms}, before {expiry_ms}")]
    PublishedBeforeExpiry {
        /// The contract settled.
        instrument: String,
        /// Its expiry.
        expiry_ms: u64,
        /// When the price was published.
        published_ms: u64,
    },
    /// Two contracts of a calendar with the same identifier.
    #[error("{instrument} appears twice in a calendar")]
    DuplicateContract {
        /// The contract.
        instrument: String,
    },
    /// Two contracts of a calendar that are not the same product.
    #[error("{instrument} is not on the same product as {first}")]
    OtherProduct {
        /// The first contract of the calendar.
        first: String,
        /// The other contract.
        instrument: String,
    },
    /// A formula of the specification refused its inputs.
    #[error(transparent)]
    Instrument(#[from] InstrumentError),
    /// The ledger's instrument is not this contract.
    #[cfg(feature = "margin")]
    #[error("the ledger's {instrument} has other terms than the specification")]
    OtherTermsInLedger {
        /// The contract.
        instrument: String,
    },
    /// The ledger refused the settlement, and is unchanged.
    #[cfg(feature = "margin")]
    #[error(transparent)]
    Ledger(#[from] MarginError),
}

/// A dated future, how it settles and who publishes its settlement price.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FutureTerms {
    spec: InstrumentSpec,
    expiry_ms: u64,
    method: SettlementMethod,
    source: SettlementSource,
}

impl FutureTerms {
    /// Terms of the dated future `spec`.
    ///
    /// # Errors
    ///
    /// [`FutureError::NotAFuture`] for any other contract.
    pub fn new(
        spec: InstrumentSpec,
        method: SettlementMethod,
        source: SettlementSource,
    ) -> Result<Self, FutureError> {
        let expiry_ms = match (spec.kind(), spec.expiry_ms()) {
            (ContractKind::Future, Some(expiry_ms)) => expiry_ms,
            (kind, _) => {
                return Err(FutureError::NotAFuture {
                    instrument: spec.id().as_str().to_string(),
                    kind,
                });
            }
        };
        Ok(Self {
            spec,
            expiry_ms,
            method,
            source,
        })
    }

    /// The contract.
    #[must_use]
    pub fn spec(&self) -> &InstrumentSpec {
        &self.spec
    }

    /// The expiry, in milliseconds since the epoch.
    #[must_use]
    pub fn expiry_ms(&self) -> u64 {
        self.expiry_ms
    }

    /// How it settles.
    #[must_use]
    pub fn method(&self) -> SettlementMethod {
        self.method
    }

    /// Who publishes its settlement price.
    #[must_use]
    pub fn source(&self) -> &SettlementSource {
        &self.source
    }

    /// The phase at `now_ms` by the clock alone, which never says
    /// [`FuturePhase::Settled`].
    #[must_use]
    pub fn phase(&self, now_ms: u64) -> FuturePhase {
        if now_ms < self.expiry_ms {
            FuturePhase::Trading
        } else {
            FuturePhase::AwaitingSettlement
        }
    }

    /// Checks what `price` declares against these terms. It cannot prove where the
    /// price came from; see the [module documentation](self).
    ///
    /// # Errors
    ///
    /// [`FutureError::WrongInstrument`], [`FutureError::WrongSource`],
    /// [`FutureError::WrongExpiry`] or [`FutureError::PublishedBeforeExpiry`].
    pub fn check_settlement(&self, price: &SettlementPrice) -> Result<(), FutureError> {
        let instrument = || self.spec.id().as_str().to_string();
        if price.instrument != *self.spec.id() {
            return Err(FutureError::WrongInstrument {
                instrument: instrument(),
                given: price.instrument.as_str().to_string(),
            });
        }
        if price.source != self.source {
            return Err(FutureError::WrongSource {
                instrument: instrument(),
                expected: self.source.as_str().to_string(),
                given: price.source.as_str().to_string(),
            });
        }
        if price.expiry_ms != self.expiry_ms {
            return Err(FutureError::WrongExpiry {
                instrument: instrument(),
                expected: self.expiry_ms,
                given: price.expiry_ms,
            });
        }
        if price.published_ms < self.expiry_ms {
            return Err(FutureError::PublishedBeforeExpiry {
                instrument: instrument(),
                expiry_ms: self.expiry_ms,
                published_ms: price.published_ms,
            });
        }
        Ok(())
    }

    /// The cash settlement of a position of `contracts` entered at `entry`.
    ///
    /// # Errors
    ///
    /// As [`FutureTerms::check_settlement`], and [`FutureError::Instrument`] for
    /// contracts or an entry that are not positive, or a result that is not exact.
    pub fn cash_settlement(
        &self,
        side: Side,
        contracts: Decimal,
        entry: Decimal,
        price: &SettlementPrice,
    ) -> Result<CashSettlement, FutureError> {
        self.check_settlement(price)?;
        match self.method {
            SettlementMethod::Cash => Ok(CashSettlement {
                pnl: self
                    .spec
                    .realized_pnl(side, contracts, entry, price.price)?,
                price: price.price,
                expiry_ms: price.expiry_ms,
                published_ms: price.published_ms,
            }),
        }
    }

    /// The ledger's identifier of this contract. An instrument it holds under that
    /// identifier with other terms, at its registered scales, is an error.
    #[cfg(feature = "margin")]
    pub(crate) fn identify_in_ledger(
        &self,
        ledger: &Ledger,
    ) -> Result<margin_rs::InstrumentId, FutureError> {
        identify_spec_in_ledger(&self.spec, ledger)
    }

    /// Books the settlement in `ledger`, closing its position at the price, at the
    /// price's publication time. A flat position settles with no cash.
    ///
    /// Booking the same price again changes nothing and returns `is_new` false. A
    /// different price, or another publication time, is refused: settlement is
    /// terminal.
    ///
    /// # Errors
    ///
    /// As [`FutureTerms::check_settlement`];
    /// [`FutureError::OtherTermsInLedger`] if the ledger's instrument is not this
    /// contract; and [`FutureError::Ledger`], with the ledger unchanged, such as for
    /// a price with more digits than the instrument's `price_scale`.
    #[cfg(feature = "margin")]
    pub fn settle_in_ledger(
        &self,
        ledger: &mut Ledger,
        price: &SettlementPrice,
    ) -> Result<PostingBooking<ExpiryCashflow>, FutureError> {
        let settlement = self.expiry_settlement(ledger, price)?;
        Ok(ledger.settle_expiry(&settlement)?)
    }

    /// As [`FutureTerms::settle_in_ledger`], in the ledger of a risk gate.
    ///
    /// # Errors
    ///
    /// As [`FutureTerms::settle_in_ledger`].
    #[cfg(feature = "margin")]
    pub fn settle_in_gate(
        &self,
        gate: &mut margin_rs::risk::RiskGate,
        price: &SettlementPrice,
    ) -> Result<PostingBooking<ExpiryCashflow>, FutureError> {
        let settlement = self.expiry_settlement(gate.ledger(), price)?;
        Ok(gate.settle_expiry(&settlement)?)
    }

    /// What a ledger books for this price, once the price and the ledger's contract
    /// are checked.
    #[cfg(feature = "margin")]
    fn expiry_settlement(
        &self,
        ledger: &Ledger,
        price: &SettlementPrice,
    ) -> Result<ExpirySettlement, FutureError> {
        self.check_settlement(price)?;
        let id = self.identify_in_ledger(ledger)?;
        let instrument = self.spec.id().as_str();
        match self.method {
            SettlementMethod::Cash => {}
        }
        Ok(ExpirySettlement {
            id: margin_rs::ExecutionId::new(format!("expiry:{instrument}"))?,
            instrument: id,
            price: price.price,
            as_of_ms: price.published_ms,
        })
    }

    /// The phase at `now_ms`: [`FuturePhase::Settled`] once `ledger` holds the
    /// settlement of this contract, and otherwise [`FutureTerms::phase`], also for a
    /// ledger without the instrument.
    ///
    /// # Errors
    ///
    /// [`FutureError::OtherTermsInLedger`] if the ledger has an instrument with this
    /// identifier and other terms, whose settlement says nothing about this contract,
    /// and [`FutureError::Ledger`] if the identifier is not valid in a ledger.
    #[cfg(feature = "margin")]
    pub fn phase_in_ledger(
        &self,
        now_ms: u64,
        ledger: &Ledger,
    ) -> Result<FuturePhase, FutureError> {
        let id = self.identify_in_ledger(ledger)?;
        Ok(if ledger.is_settled(&id) {
            FuturePhase::Settled
        } else {
            self.phase(now_ms)
        })
    }
}

/// Declared futures on one product, by expiry, then by identifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FutureCalendar(Vec<FutureTerms>);

impl FutureCalendar {
    /// A calendar of `contracts`, which must share their underlying, price asset
    /// and settlement asset. Two contracts may share an expiry, such as a
    /// weekly and a monthly: the identifier then orders them.
    ///
    /// # Errors
    ///
    /// [`FutureError::DuplicateContract`] or [`FutureError::OtherProduct`].
    pub fn new(mut contracts: Vec<FutureTerms>) -> Result<Self, FutureError> {
        contracts.sort_by(|a, b| {
            (a.expiry_ms, a.spec.id().as_str()).cmp(&(b.expiry_ms, b.spec.id().as_str()))
        });
        let mut ids: Vec<&str> = contracts.iter().map(|c| c.spec.id().as_str()).collect();
        ids.sort_unstable();
        if let Some(pair) = ids.windows(2).find(|pair| pair.first() == pair.get(1)) {
            return Err(FutureError::DuplicateContract {
                instrument: pair.first().copied().unwrap_or_default().to_string(),
            });
        }
        if let Some(first) = contracts.first()
            && let Some(other) = contracts.iter().find(|c| {
                c.spec.underlying() != first.spec.underlying()
                    || c.spec.price_asset() != first.spec.price_asset()
                    || c.spec.settlement_asset() != first.spec.settlement_asset()
            })
        {
            return Err(FutureError::OtherProduct {
                first: first.spec.id().as_str().to_string(),
                instrument: other.spec.id().as_str().to_string(),
            });
        }
        Ok(Self(contracts))
    }

    /// The contracts, by expiry, then by identifier.
    #[must_use]
    pub fn contracts(&self) -> &[FutureTerms] {
        &self.0
    }

    /// The contracts still trading at `now_ms`, in the same order.
    pub fn trading_at(&self, now_ms: u64) -> impl Iterator<Item = &FutureTerms> {
        self.0.iter().filter(move |c| c.expiry_ms > now_ms)
    }
}

/// The ledger's identifier of `spec`. An instrument it holds under that identifier with
/// other terms, at its registered scales, is an error.
#[cfg(feature = "margin")]
pub(crate) fn identify_spec_in_ledger(
    spec: &InstrumentSpec,
    ledger: &Ledger,
) -> Result<margin_rs::InstrumentId, FutureError> {
    let instrument = spec.id().as_str();
    let id = margin_rs::InstrumentId::new(instrument)?;
    if let Some(registered) = ledger.instrument(&id) {
        let same = super::margin::registered_scales(registered)
            .and_then(|scales| margin_rs::Instrument::from_spec(spec, scales).ok())
            .is_some_and(|expected| expected == *registered);
        if !same {
            return Err(FutureError::OtherTermsInLedger {
                instrument: instrument.to_string(),
            });
        }
    }
    Ok(id)
}
