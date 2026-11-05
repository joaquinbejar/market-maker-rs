//! A plan to roll a position from a dated future to a later one, in two legs, and what
//! may be quoted and reserved meanwhile ([`RollGuard`]).
//!
//! A [`RollPlan`] closes a position on the near contract and opens it on the far
//! one. It only emits intents; sending orders is the caller's job, and execution is
//! never assumed to be atomic, so one leg can fill alone. With `margin` too, `run` runs
//! it through `backtest::contracts::ContractEngine`.
//!
//! # Exposure
//!
//! What a roll preserves is `contracts * size`, in the asset of the contract's size,
//! with a sign:
//!
//! - on a linear contract, units of the underlying, at any price;
//! - on an inverse contract, value in the contract currency, such as USD. Its delta
//!   in the underlying, `contracts * value / price`, moves with the price, so it is
//!   not what is preserved, and it is not reported here.
//!
//! Both legs must be of one product ([`FutureCalendar`]). `exposure_gap` in a report
//! is the exposure now minus the exposure when the plan was made: after one leg alone
//! it is not zero, which is the point.
//!
//! # What executed
//!
//! The plan keeps every observed fill itself, and positions, exposure and fees come
//! only from that record. A real fill is never refused for what it does to the plan:
//! only one that is not a fill at all, with a price or a quantity that is not
//! positive, an identifier reused with other content, or a position or exposure that
//! a `Decimal` cannot hold, is an error.
//!
//! For the intents (cancel, resubmit, unwind), `arbitrage_rs::package::PackagePlan`
//! serves as a primitive of quantities. It sees neutral economics, a price of one and
//! no fee, and is never read for cash. It is stricter than what can happen, refusing
//! for example a fill off the lot or on a rejected attempt. When it refuses a fill,
//! the fill stays recorded and positions stay right, the report says why, the state
//! becomes [`RollState::Faulted`] and every live attempt is cancelled once. From then
//! on only the record moves, and the caller decides.
//!
//! On an inverse contract the primitive's `contract_size` is given the contract value
//! rather than units of the underlying. Its grid, matched package and excess only
//! multiply and compare, so they are in value there; a test pins this.
//!
//! # Time
//!
//! A deadline stops new orders, not what already executed. It may not be after the
//! near contract's expiry, so nothing is resubmitted or opened beyond it.
//!
//! [`RollEvent::NearSettled`] declares that the near contract settled some contracts
//! at a [`SettlementPrice`]: they leave its position, and the gap shows what was lost.
//! The plan never settles anything itself, which `FutureTerms::settle_in_ledger` does.
//!
//! A near fill observed after that is one of three things, as in a `margin-rs` ledger:
//!
//! - Executed at or after the expiry, a ledger refuses it. It moves the position like
//!   any fill, so a residual shows that something is wrong.
//! - Executed before the expiry and already in the declared settlement's position, the
//!   ledger booked it first and closed fewer contracts, so a residual was left on the
//!   plan's side. It moves the position back to zero.
//! - Executed before the expiry beyond that, a ledger books it after the settlement
//!   and closes it at the settlement price at once, so the position does not move, and
//!   the contracts are in [`RollPlan::closed_at_settlement`].
//!
//! Which of the last two it was is inferred from the settlement's declared contracts,
//! so it can be wrong, for example after a fill outside the plan. The record of fills is
//! not: a reconciliation compares the closed quantity of each fill with a ledger.

use std::collections::BTreeMap;

use arbitrage_rs::legs::{ExecId, FailurePolicy, Fallback, LegSpec};
use arbitrage_rs::package::{PackageAction, PackageEvent, PackageLegSpec, PackagePlan};
use arbitrage_rs::venues::VenueId;
use arbitrage_rs::{ArbitrageError, Cost, Currency, Leg, Money, Price, Quantity, Timestamp};
use financial_types::Action;
use rust_decimal::Decimal;

pub mod guard;
#[cfg(feature = "margin")]
pub mod run;

pub use guard::RollGuard;
#[cfg(feature = "margin")]
pub use guard::RollReserves;

use super::AssetCode;
use super::futures::{FutureCalendar, FutureError, FuturePhase, FutureTerms, SettlementPrice};

/// The leg of a roll.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RollLeg {
    /// The contract being closed.
    Near,
    /// The contract being opened.
    Far,
}

/// What to do after a recovery by resubmission is used up.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RollFallback {
    /// Cancel the other leg and keep the gap.
    Abort,
    /// Cancel the other leg, then ask to offset the excess.
    Unwind,
}

/// How the plan reacts when a leg fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RollFailurePolicy {
    /// Cancel the other leg and keep the gap.
    Abort,
    /// Cancel the other leg, then ask to offset the excess.
    Unwind,
    /// Resubmit the failed leg up to `max_resubmits` times over the plan's life and
    /// never past the deadline, then apply `then`.
    Complete {
        /// Resubmissions allowed.
        max_resubmits: u32,
        /// What to do once they are used up.
        then: RollFallback,
    },
}

/// Input to [`RollPlan::new`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RollRequest {
    /// The contract to close.
    pub near: FutureTerms,
    /// The contract to open, expiring after `near`, of the same product.
    pub far: FutureTerms,
    /// The position on `near`, signed. It may be off the lot: what is off the grid
    /// stays open there, as [`RollPlan::uncovered`].
    pub near_position: Decimal,
    /// The position already on `far`, signed; it may be of either sign.
    pub far_position: Decimal,
    /// The limit price of the near leg.
    pub near_price: Decimal,
    /// The limit price of the far leg.
    pub far_price: Decimal,
    /// How to react when a leg fails.
    pub policy: RollFailurePolicy,
    /// When to stop working the legs: after `created_at_ms`, and not after the near
    /// contract's expiry.
    pub deadline_ms: u64,
    /// When the plan is made, before the near contract's expiry.
    pub created_at_ms: u64,
}

/// One observed execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RollFill {
    /// The leg.
    pub leg: RollLeg,
    /// The order attempt it belongs to, from 0 and as given by [`RollAction`]. `None`
    /// for a fill outside the plan's attempts, such as an unwind's: it is recorded,
    /// but not given to the plan's intents.
    pub attempt: Option<u32>,
    /// The venue's execution identifier, unique within the leg's contract.
    pub exec_id: String,
    /// The side.
    pub side: Action,
    /// The contracts filled.
    pub contracts: Decimal,
    /// The attempt's cumulative contracts, this fill included.
    pub cum_contracts: Decimal,
    /// The price.
    pub price: Decimal,
    /// The fee: negative is a rebate.
    pub fee: Decimal,
    /// The asset of the fee.
    pub fee_asset: AssetCode,
    /// When it executed, in milliseconds since the epoch.
    pub at_ms: u64,
}

/// An observed event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RollEvent {
    /// A fill.
    Fill(RollFill),
    /// An attempt was rejected without filling.
    Rejected {
        /// The leg.
        leg: RollLeg,
        /// The attempt.
        attempt: u32,
        /// When it was observed.
        at_ms: u64,
    },
    /// An attempt's open quantity is cancelled.
    CancelAck {
        /// The leg.
        leg: RollLeg,
        /// The attempt.
        attempt: u32,
        /// Its final cumulative contracts.
        cum_contracts: Decimal,
        /// When it was observed.
        at_ms: u64,
    },
    /// Time has passed with no execution, which checks the deadline.
    Tick {
        /// The time.
        at_ms: u64,
    },
    /// The near contract settled these contracts at this price.
    NearSettled {
        /// The settlement price, with what it declares.
        price: SettlementPrice,
        /// The contracts closed, signed like the position; zero if it was flat.
        contracts: Decimal,
    },
}

/// An intent for the caller to carry out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RollAction {
    /// Cancel an attempt's open quantity.
    Cancel {
        /// The leg.
        leg: RollLeg,
        /// The attempt.
        attempt: u32,
    },
    /// Send the leg again as a new attempt.
    Resubmit {
        /// The leg.
        leg: RollLeg,
        /// The new attempt.
        attempt: u32,
        /// The contracts still to execute.
        contracts: Decimal,
        /// The plan's limit price for the leg.
        price: Decimal,
    },
    /// Offset a leg's excess.
    Unwind {
        /// The leg.
        leg: RollLeg,
        /// The side that reduces it.
        side: Action,
        /// The contracts to offset.
        contracts: Decimal,
    },
}

/// Where the plan stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum RollState {
    /// Both legs are being worked.
    Working,
    /// Both legs are executed in full.
    Completed,
    /// A leg failed, and the other is being cancelled before aborting.
    Aborting,
    /// A leg failed, and the other is being cancelled before unwinding.
    Unwinding,
    /// Both legs are done, and the gap is kept.
    Aborted,
    /// Both legs are done, and each excess is asked to be offset.
    UnwindRequested,
    /// The intents stopped, because a real fill did not fit them: see
    /// [`RollPlan::fault`]. Only the record of fills moves now.
    Faulted,
}

/// How far the legs match, in the asset of the contract's size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RollProgress {
    /// The package executed on both legs.
    pub matched: Decimal,
    /// The near leg's contracts beyond it.
    pub near_excess: Decimal,
    /// The far leg's contracts beyond it.
    pub far_excess: Decimal,
}

/// What [`RollPlan::apply`] returns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RollReport {
    /// Intents to carry out, in order.
    pub actions: Vec<RollAction>,
    /// Where the plan stands after the event.
    pub state: RollState,
    /// The position on the near contract.
    pub near_position: Decimal,
    /// The position on the far contract.
    pub far_position: Decimal,
    /// The signed exposure over both legs.
    pub exposure: Decimal,
    /// The exposure minus the exposure when the plan was made.
    pub exposure_gap: Decimal,
    /// The event was a repeat, and changed nothing.
    pub duplicate: bool,
    /// Why the plan's intents did not take this fill, which is recorded anyway.
    pub refused: Option<String>,
}

/// Why a roll or an event was refused. The plan is unchanged by an error.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum RollError {
    /// The contracts' own terms, or a ledger's.
    #[error(transparent)]
    Future(#[from] FutureError),
    /// The far contract does not expire after the near one.
    #[error("{far} does not expire after {near}")]
    FarNotLater {
        /// The near contract.
        near: String,
        /// The far contract.
        far: String,
    },
    /// The near contract is no longer trading.
    #[error("{near} expired at {expiry_ms}, at or before the plan at {created_at_ms}")]
    NearExpired {
        /// The near contract.
        near: String,
        /// Its expiry.
        expiry_ms: u64,
        /// When the plan was made.
        created_at_ms: u64,
    },
    /// A deadline that is not after the plan, or is after the near contract's expiry.
    #[error("deadline {deadline_ms} is not after {created_at_ms} and by {expiry_ms}")]
    InvalidDeadline {
        /// The deadline.
        deadline_ms: u64,
        /// When the plan was made.
        created_at_ms: u64,
        /// The near contract's expiry.
        expiry_ms: u64,
    },
    /// Nothing of the position fits the grid of both contracts.
    #[error("exposure {exposure} is below the grid {grid} of both contracts")]
    NothingToRoll {
        /// The absolute exposure on the near contract.
        exposure: Decimal,
        /// The grid.
        grid: Decimal,
    },
    /// A limit price that is not positive.
    #[error("limit price {price} is not positive")]
    InvalidPrice {
        /// The price.
        price: Decimal,
    },
    /// A fill that is not one.
    #[error("not a fill: {reason}")]
    InvalidFill {
        /// Why not.
        reason: String,
    },
    /// An execution identifier reused on its leg with other content.
    #[error("execution {exec_id} was observed with other content")]
    ConflictingFill {
        /// The identifier.
        exec_id: String,
    },
    /// A settlement reported again with other data.
    #[error("the near contract's settlement was observed with other data")]
    ConflictingSettlement,
    /// An exposure limit below the exposure already held.
    #[error("exposure limit {max_exposure} is below the exposure of {exposure}")]
    InvalidExposureLimit {
        /// The limit.
        max_exposure: Decimal,
        /// The exposure now, in absolute value.
        exposure: Decimal,
    },
    /// A guard given another plan, by anything in its request.
    #[error("the plan is not the one the guard was made for")]
    OtherPlan,
    /// A guard given a plan that does not descend from what an earlier sync saw.
    #[error("the plan does not descend from what an earlier sync saw")]
    StalePlan,
    /// A reservation on a near contract that settled.
    #[cfg(feature = "margin")]
    #[error("the near contract settled")]
    NearAlreadySettled,
    /// A settlement reported of a near contract that a ledger has not settled.
    #[cfg(feature = "margin")]
    #[error("the ledger has not settled the near contract")]
    NearNotSettled,
    /// A reconciliation of a plan that was not anchored to its ledger.
    #[cfg(feature = "margin")]
    #[error("the plan was not anchored to a ledger: see RollPlan::anchored_to")]
    NotAnchored,
    /// A time beyond what the plan can represent.
    #[error("{field} is {value}, beyond the largest time supported")]
    TimeOutOfRange {
        /// Which one.
        field: &'static str,
        /// What it was.
        value: u64,
    },
    /// The plan's intents refused an event that is not a fill, or a figure overflowed.
    #[error("roll plan: {reason}")]
    Plan {
        /// The reason.
        reason: String,
    },
}

/// Private, so that no `arbitrage-rs` type is in this crate's public API.
fn upstream(error: ArbitrageError) -> RollError {
    RollError::Plan {
        reason: error.to_string(),
    }
}

fn time(field: &'static str, value: u64) -> Result<Timestamp, RollError> {
    i64::try_from(value)
        .map(Timestamp::from_millis)
        .map_err(|_| RollError::TimeOutOfRange { field, value })
}

fn exact_mul(a: Decimal, b: Decimal) -> Result<Decimal, RollError> {
    arbitrage_rs::exact::mul(a, b).map_err(upstream)
}

fn exact_add(a: Decimal, b: Decimal) -> Result<Decimal, RollError> {
    arbitrage_rs::exact::add(a, b).map_err(upstream)
}

fn exact_sub(a: Decimal, b: Decimal) -> Result<Decimal, RollError> {
    arbitrage_rs::exact::sub(a, b).map_err(upstream)
}

/// The least common multiple of two positive values, exactly.
fn least_common_multiple(a: Decimal, b: Decimal) -> Result<Decimal, RollError> {
    let overflow = || RollError::Plan {
        reason: "the grid of both contracts does not fit".to_string(),
    };
    let (a, b) = (a.normalize(), b.normalize());
    let scale = a.scale().max(b.scale());
    let up = |d: Decimal| {
        10_i128
            .checked_pow(scale.saturating_sub(d.scale()))
            .and_then(|p| d.mantissa().checked_mul(p))
            .ok_or_else(overflow)
    };
    let (x, y) = (up(a)?, up(b)?);
    let (mut m, mut n) = (x, y);
    while n != 0 {
        (m, n) = (n, m.checked_rem(n).ok_or_else(overflow)?);
    }
    let lcm = x
        .checked_div(m)
        .and_then(|q| q.checked_mul(y))
        .ok_or_else(overflow)?;
    Decimal::try_from_i128_with_scale(lcm, scale)
        .map(|d| d.normalize())
        .map_err(|_| overflow())
}

fn quantity(value: Decimal) -> Result<Quantity, RollError> {
    Quantity::new(value).map_err(upstream)
}

/// One leg's constants.
#[derive(Debug, Clone, PartialEq, Eq)]
struct LegTerms {
    terms: FutureTerms,
    size: Decimal,
    side: Action,
    price: Decimal,
    contracts: Decimal,
    start: Decimal,
}

/// A recorded settlement of the near contract.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Settled {
    price: SettlementPrice,
    contracts: Decimal,
}

/// A two contract roll, driven by observed events.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RollPlan {
    near: LegTerms,
    far: LegTerms,
    package: Decimal,
    uncovered: Decimal,
    grid: Decimal,
    policy: RollFailurePolicy,
    created_at_ms: u64,
    deadline_ms: u64,
    start_exposure: Decimal,
    near_position: Decimal,
    far_position: Decimal,
    fills: Vec<RollFill>,
    settled: Option<Settled>,
    closed_at_settlement: Decimal,
    /// Near fills with a part closed at the settlement, and that part, signed.
    closed_fills: Vec<(String, Decimal)>,
    /// The events that changed the plan, so that a copy can be told to descend from it.
    history: Vec<RollEvent>,
    /// The latest time in any event, from the plan's creation on.
    observed_ms: u64,
    /// The length of a ledger's journal when the plan was anchored to it.
    #[cfg(feature = "margin")]
    journal_anchor: Option<u64>,
    intents: PackagePlan,
    fault: Option<String>,
}

const NEAR_LEG: usize = 0;
const FAR_LEG: usize = 1;

fn index(leg: RollLeg) -> usize {
    match leg {
        RollLeg::Near => NEAR_LEG,
        RollLeg::Far => FAR_LEG,
    }
}

fn leg_at(index: usize) -> RollLeg {
    if index == NEAR_LEG {
        RollLeg::Near
    } else {
        RollLeg::Far
    }
}

fn neutral_price() -> Result<Price, RollError> {
    Price::new(Decimal::ONE).map_err(upstream)
}

fn neutral_currency() -> Result<Currency, RollError> {
    Currency::new("UNIT").map_err(upstream)
}

impl RollPlan {
    /// Plans a roll.
    ///
    /// # Errors
    ///
    /// [`RollError`] for contracts that cannot be rolled, a bad time or price, or a
    /// position with nothing on the grid of both contracts.
    pub fn new(request: RollRequest) -> Result<Self, RollError> {
        let RollRequest {
            near,
            far,
            near_position,
            far_position,
            near_price,
            far_price,
            policy,
            deadline_ms,
            created_at_ms,
        } = request;
        FutureCalendar::new(vec![near.clone(), far.clone()])?;
        let id = |terms: &FutureTerms| terms.spec().id().as_str().to_string();
        if far.expiry_ms() <= near.expiry_ms() {
            return Err(RollError::FarNotLater {
                near: id(&near),
                far: id(&far),
            });
        }
        if near.phase(created_at_ms) != FuturePhase::Trading {
            return Err(RollError::NearExpired {
                near: id(&near),
                expiry_ms: near.expiry_ms(),
                created_at_ms,
            });
        }
        if deadline_ms <= created_at_ms || deadline_ms > near.expiry_ms() {
            return Err(RollError::InvalidDeadline {
                deadline_ms,
                created_at_ms,
                expiry_ms: near.expiry_ms(),
            });
        }
        for price in [near_price, far_price] {
            if price <= Decimal::ZERO {
                return Err(RollError::InvalidPrice { price });
            }
        }
        let created_at = time("created_at_ms", created_at_ms)?;
        let deadline = time("deadline_ms", deadline_ms)?;
        let near_expiry = time("expiry_ms", near.expiry_ms())?;
        let far_expiry = time("expiry_ms", far.expiry_ms())?;

        let (near_size, far_size) = (near.spec().size().amount(), far.spec().size().amount());
        let near_exposure = exact_mul(near_position, near_size)?;
        let far_exposure = exact_mul(far_position, far_size)?;
        let start_exposure = exact_add(near_exposure, far_exposure)?;
        let lot_of = |terms: &FutureTerms, size| exact_mul(terms.spec().lot(), size);
        let grid = least_common_multiple(lot_of(&near, near_size)?, lot_of(&far, far_size)?)?;
        let absolute = near_exposure.abs();
        let nothing = || RollError::NothingToRoll {
            exposure: absolute,
            grid,
        };
        let package = match Quantity::new(absolute) {
            Ok(exposure) => exposure
                .floor_to_lot(quantity(grid)?)
                .map_err(upstream)?
                .ok_or_else(nothing)?,
            Err(_) => return Err(nothing()),
        };
        let uncovered = exact_sub(absolute, package.value())?;

        let contracts_of = |size: Decimal| -> Result<Decimal, RollError> {
            let contracts = package
                .value()
                .checked_div(size)
                .ok_or_else(|| RollError::Plan {
                    reason: "the package in contracts overflows".to_string(),
                })?;
            if exact_mul(contracts, size)? == package.value() {
                Ok(contracts.normalize())
            } else {
                Err(RollError::Plan {
                    reason: "the package is not exact in contracts".to_string(),
                })
            }
        };
        let (near_contracts, far_contracts) = (contracts_of(near_size)?, contracts_of(far_size)?);
        // The far side opens what the near side closes.
        let (near_side, far_side) = if near_position > Decimal::ZERO {
            (Action::Sell, Action::Buy)
        } else {
            (Action::Buy, Action::Sell)
        };
        let venue = VenueId::new("roll").map_err(upstream)?;
        let spec_of = |terms: &FutureTerms,
                       side,
                       contracts,
                       size,
                       expiry|
         -> Result<PackageLegSpec, RollError> {
            Ok(PackageLegSpec {
                spec: LegSpec {
                    leg: Leg {
                        instrument: arbitrage_rs::Instrument::dated(
                            arbitrage_rs::InstrumentId::new(terms.spec().id().as_str())
                                .map_err(upstream)?,
                            expiry,
                        ),
                        side,
                        quantity: quantity(contracts)?,
                        price: neutral_price()?,
                    },
                    deadline: Some(deadline),
                    venue: venue.clone(),
                },
                contract_size: quantity(size)?,
                lot: quantity(terms.spec().lot())?,
            })
        };
        let intents = PackagePlan::new(
            vec![
                spec_of(&near, near_side, near_contracts, near_size, near_expiry)?,
                spec_of(&far, far_side, far_contracts, far_size, far_expiry)?,
            ],
            package,
            match policy {
                RollFailurePolicy::Abort => FailurePolicy::Abort,
                RollFailurePolicy::Unwind => FailurePolicy::Unwind,
                RollFailurePolicy::Complete {
                    max_resubmits,
                    then,
                } => FailurePolicy::Complete {
                    max_resubmits,
                    then: match then {
                        RollFallback::Abort => Fallback::Abort,
                        RollFallback::Unwind => Fallback::Unwind,
                    },
                },
            },
            neutral_currency()?,
            created_at,
        )
        .map_err(upstream)?;
        Ok(Self {
            near: LegTerms {
                terms: near,
                size: near_size,
                side: near_side,
                price: near_price,
                contracts: near_contracts,
                start: near_position,
            },
            far: LegTerms {
                terms: far,
                size: far_size,
                side: far_side,
                price: far_price,
                contracts: far_contracts,
                start: far_position,
            },
            package: package.value(),
            uncovered,
            grid,
            policy,
            created_at_ms,
            deadline_ms,
            start_exposure,
            near_position,
            far_position,
            fills: Vec::new(),
            settled: None,
            closed_at_settlement: Decimal::ZERO,
            closed_fills: Vec::new(),
            history: Vec::new(),
            observed_ms: created_at_ms,
            #[cfg(feature = "margin")]
            journal_anchor: None,
            intents,
            fault: None,
        })
    }

    fn leg(&self, leg: RollLeg) -> &LegTerms {
        match leg {
            RollLeg::Near => &self.near,
            RollLeg::Far => &self.far,
        }
    }

    /// The position on a leg now.
    #[must_use]
    pub fn position(&self, leg: RollLeg) -> Decimal {
        match leg {
            RollLeg::Near => self.near_position,
            RollLeg::Far => self.far_position,
        }
    }

    /// The contracts the plan trades on a leg.
    #[must_use]
    pub fn contracts(&self, leg: RollLeg) -> Decimal {
        self.leg(leg).contracts
    }

    /// The side the plan trades on a leg.
    #[must_use]
    pub fn side(&self, leg: RollLeg) -> Action {
        self.leg(leg).side
    }

    /// The package, in the asset of the contract's size.
    #[must_use]
    pub fn package(&self) -> Decimal {
        self.package
    }

    /// The grid of both contracts' lots, in the asset of the contract's size.
    #[must_use]
    pub fn grid(&self) -> Decimal {
        self.grid
    }

    /// The exposure off the grid, which the plan leaves on the near contract.
    #[must_use]
    pub fn uncovered(&self) -> Decimal {
        self.uncovered
    }

    /// The signed exposure over both legs now.
    ///
    /// # Errors
    ///
    /// [`RollError::Plan`] if it overflows.
    pub fn exposure(&self) -> Result<Decimal, RollError> {
        exposure(
            self.near_position,
            self.near.size,
            self.far_position,
            self.far.size,
        )
    }

    /// The exposure now minus the exposure when the plan was made.
    ///
    /// # Errors
    ///
    /// [`RollError::Plan`] if it overflows.
    pub fn exposure_gap(&self) -> Result<Decimal, RollError> {
        exact_sub(self.exposure()?, self.start_exposure)
    }

    /// Every fill observed, in order.
    #[must_use]
    pub fn fills(&self) -> &[RollFill] {
        &self.fills
    }

    /// The contracts the near contract settled, if it did.
    #[must_use]
    pub fn near_settled(&self) -> Option<Decimal> {
        self.settled.as_ref().map(|s| s.contracts)
    }

    /// The signed contracts of near fills executed before the expiry and observed
    /// after the settlement, which closed at the settlement price: see the
    /// [module documentation](self).
    #[must_use]
    pub fn closed_at_settlement(&self) -> Decimal {
        self.closed_at_settlement
    }

    /// The fees of every fill observed, per asset, never summed across assets.
    ///
    /// # Errors
    ///
    /// [`RollError::Plan`] if a sum overflows.
    pub fn fees(&self) -> Result<BTreeMap<AssetCode, Decimal>, RollError> {
        let mut fees = BTreeMap::new();
        for fill in &self.fills {
            let sum = fees.entry(fill.fee_asset.clone()).or_insert(Decimal::ZERO);
            *sum = exact_add(*sum, fill.fee)?;
        }
        Ok(fees)
    }

    /// Why the intents stopped, if they did.
    #[must_use]
    pub fn fault(&self) -> Option<&str> {
        self.fault.as_deref()
    }

    /// Where the plan stands.
    #[must_use]
    pub fn state(&self) -> RollState {
        if self.fault.is_some() {
            return RollState::Faulted;
        }
        use arbitrage_rs::legs::PlanState;
        match self.intents.state() {
            PlanState::Working => RollState::Working,
            PlanState::Completed => RollState::Completed,
            PlanState::Aborting => RollState::Aborting,
            PlanState::Unwinding => RollState::Unwinding,
            PlanState::Aborted => RollState::Aborted,
            PlanState::UnwindRequested => RollState::UnwindRequested,
        }
    }

    /// How far the legs match, or `None` once faulted, when only the record is right.
    ///
    /// # Errors
    ///
    /// [`RollError::Plan`] if a figure overflows.
    pub fn progress(&self) -> Result<Option<RollProgress>, RollError> {
        if self.fault.is_some() {
            return Ok(None);
        }
        let excess = self.intents.excess().map_err(upstream)?;
        let of = |i: usize| excess.get(i).copied().unwrap_or(Decimal::ZERO);
        Ok(Some(RollProgress {
            matched: self.intents.matched().map_err(upstream)?,
            near_excess: of(NEAR_LEG),
            far_excess: of(FAR_LEG),
        }))
    }

    fn report(
        &self,
        actions: Vec<RollAction>,
        duplicate: bool,
        refused: Option<String>,
    ) -> Result<RollReport, RollError> {
        Ok(RollReport {
            actions,
            state: self.state(),
            near_position: self.near_position,
            far_position: self.far_position,
            exposure: self.exposure()?,
            exposure_gap: self.exposure_gap()?,
            duplicate,
            refused,
        })
    }

    fn map_action(&self, action: &PackageAction) -> RollAction {
        match action {
            PackageAction::Cancel { leg, attempt } => RollAction::Cancel {
                leg: leg_at(*leg),
                attempt: *attempt,
            },
            PackageAction::Resubmit {
                leg,
                attempt,
                quantity,
                ..
            } => RollAction::Resubmit {
                leg: leg_at(*leg),
                attempt: *attempt,
                contracts: quantity.value(),
                price: self.leg(leg_at(*leg)).price,
            },
            PackageAction::Unwind {
                leg,
                side,
                quantity,
                ..
            } => RollAction::Unwind {
                leg: leg_at(*leg),
                side: *side,
                contracts: quantity.value(),
            },
        }
    }

    /// Applies one observed event. An error leaves the plan unchanged.
    ///
    /// # Errors
    ///
    /// [`RollError`]: see the [module documentation](self) for what is an error for a
    /// fill. Any other event the intents refuse is one too, unless the plan is faulted.
    pub fn apply(&mut self, event: &RollEvent) -> Result<RollReport, RollError> {
        let report = self.apply_event(event)?;
        if !report.duplicate {
            self.history.push(event.clone());
            let at_ms = match event {
                RollEvent::Fill(fill) => fill.at_ms,
                RollEvent::Rejected { at_ms, .. }
                | RollEvent::CancelAck { at_ms, .. }
                | RollEvent::Tick { at_ms } => *at_ms,
                RollEvent::NearSettled { price, .. } => price.published_ms(),
            };
            self.observed_ms = self.observed_ms.max(at_ms);
        }
        Ok(report)
    }

    fn apply_event(&mut self, event: &RollEvent) -> Result<RollReport, RollError> {
        match event {
            RollEvent::Fill(fill) => self.apply_fill(fill),
            RollEvent::NearSettled { price, contracts } => self.apply_settlement(price, *contracts),
            RollEvent::Rejected {
                leg,
                attempt,
                at_ms,
            } => self.step(&PackageEvent::Rejected {
                leg: index(*leg),
                attempt: *attempt,
                at: time("at_ms", *at_ms)?,
            }),
            RollEvent::CancelAck {
                leg,
                attempt,
                cum_contracts,
                at_ms,
            } => self.step(&PackageEvent::CancelAck {
                leg: index(*leg),
                attempt: *attempt,
                cum_qty: *cum_contracts,
                at: time("at_ms", *at_ms)?,
            }),
            RollEvent::Tick { at_ms } => self.step(&PackageEvent::Tick {
                at: time("at_ms", *at_ms)?,
            }),
        }
    }

    /// Checks that a report of these positions can be made, before anything changes.
    fn check_figures(&self, near: Decimal, far: Decimal) -> Result<(), RollError> {
        let now = exposure(near, self.near.size, far, self.far.size)?;
        exact_sub(now, self.start_exposure)?;
        Ok(())
    }

    /// Gives the intents an event that is not a fill.
    fn step(&mut self, event: &PackageEvent) -> Result<RollReport, RollError> {
        if self.fault.is_some() {
            return self.report(Vec::new(), false, None);
        }
        let report = self.intents.apply(event).map_err(upstream)?;
        let actions = report.actions.iter().map(|a| self.map_action(a)).collect();
        self.report(actions, report.duplicate, None)
    }

    fn apply_settlement(
        &mut self,
        price: &SettlementPrice,
        contracts: Decimal,
    ) -> Result<RollReport, RollError> {
        self.near.terms.check_settlement(price)?;
        if let Some(settled) = &self.settled {
            if settled.price == *price && settled.contracts == contracts {
                return self.report(Vec::new(), true, None);
            }
            return Err(RollError::ConflictingSettlement);
        }
        let near = exact_sub(self.near_position, contracts)?;
        self.check_figures(near, self.far_position)?;
        // The deadline is past, so this can only ask to cancel what is open. The news may
        // arrive after the time it carries: the plan's clock does not go back, and whatever
        // the deadline asked for it already asked.
        let published = time("published_ms", price.published_ms())?;
        let mut actions = Vec::new();
        if self.fault.is_none()
            && let Ok(report) = self.intents.apply(&PackageEvent::Tick { at: published })
        {
            actions = report.actions.iter().map(|a| self.map_action(a)).collect();
        }
        self.settled = Some(Settled {
            price: price.clone(),
            contracts,
        });
        self.near_position = near;
        self.report(actions, false, None)
    }

    fn apply_fill(&mut self, fill: &RollFill) -> Result<RollReport, RollError> {
        if fill.price <= Decimal::ZERO || fill.contracts <= Decimal::ZERO {
            return Err(RollError::InvalidFill {
                reason: format!(
                    "execution {} has price {} and contracts {}",
                    fill.exec_id, fill.price, fill.contracts
                ),
            });
        }
        let signed = match fill.side {
            Action::Buy => fill.contracts,
            Action::Sell => arbitrage_rs::exact::neg(fill.contracts),
            other => {
                return Err(RollError::InvalidFill {
                    reason: format!("execution {} is on {other}", fill.exec_id),
                });
            }
        };
        // A venue's execution identifiers are per contract, so each leg has its own.
        if let Some(seen) = self
            .fills
            .iter()
            .find(|f| f.leg == fill.leg && f.exec_id == fill.exec_id)
        {
            if seen == fill {
                return self.report(Vec::new(), true, None);
            }
            return Err(RollError::ConflictingFill {
                exec_id: fill.exec_id.clone(),
            });
        }
        let (mut near, mut far) = (self.near_position, self.far_position);
        let mut closed_now = Decimal::ZERO;
        match fill.leg {
            RollLeg::Far => far = exact_add(far, signed)?,
            RollLeg::Near => {
                // Executed before the expiry and observed after the settlement: what
                // takes the position back to zero moves it, and the rest closed at the
                // settlement price at once.
                let late = self.settled.is_some() && fill.at_ms < self.near.terms.expiry_ms();
                let moves = if !late {
                    signed
                } else if near.is_sign_positive() == signed.is_sign_positive() || near.is_zero() {
                    Decimal::ZERO
                } else if near.abs() < signed.abs() {
                    arbitrage_rs::exact::neg(near)
                } else {
                    signed
                };
                near = exact_add(near, moves)?;
                closed_now = exact_sub(signed, moves)?;
            }
        }
        let closed = exact_add(self.closed_at_settlement, closed_now)?;
        self.check_figures(near, far)?;

        let mut refused = None;
        let mut actions = Vec::new();
        if self.fault.is_none()
            && let Some(attempt) = fill.attempt
        {
            match self.give_to_intents(fill, attempt) {
                Ok(given) => actions = given,
                Err(reason) => {
                    actions = self.live_attempts();
                    self.fault = Some(reason.clone());
                    refused = Some(reason);
                }
            }
        }
        self.near_position = near;
        self.far_position = far;
        self.closed_at_settlement = closed;
        if !closed_now.is_zero() {
            self.closed_fills.push((fill.exec_id.clone(), closed_now));
        }
        self.fills.push(fill.clone());
        self.report(actions, false, refused)
    }

    /// Gives the intents a fill, in neutral economics, or says why they refuse it.
    fn give_to_intents(
        &mut self,
        fill: &RollFill,
        attempt: u32,
    ) -> Result<Vec<RollAction>, String> {
        if fill.side != self.leg(fill.leg).side {
            return Err(format!(
                "execution {} is on {}, and the leg trades {}",
                fill.exec_id,
                fill.side,
                self.leg(fill.leg).side
            ));
        }
        let at = time("at_ms", fill.at_ms).map_err(|e| e.to_string())?;
        let text = |e: RollError| e.to_string();
        let contracts = quantity(fill.contracts).map_err(text)?;
        let cum_contracts = quantity(fill.cum_contracts).map_err(text)?;
        let currency = neutral_currency().map_err(text)?;
        let price = neutral_price().map_err(text)?;
        let report = self
            .intents
            .apply(&PackageEvent::Fill {
                leg: index(fill.leg),
                attempt,
                exec_id: ExecId(format!("{:?}:{}", fill.leg, fill.exec_id)),
                quantity: contracts,
                cum_qty: cum_contracts,
                price,
                fee: Cost::new(Money::zero(currency)),
                at,
            })
            .map_err(|e| e.to_string())?;
        Ok(report.actions.iter().map(|a| self.map_action(a)).collect())
    }

    /// A cancel of every attempt that is still open and was not asked to cancel.
    fn live_attempts(&self) -> Vec<RollAction> {
        let mut actions = Vec::new();
        for (i, leg) in self.intents.legs().iter().enumerate() {
            for (attempt, state) in leg.attempts.iter().enumerate() {
                if state.is_live() && !state.cancel_requested {
                    actions.push(RollAction::Cancel {
                        leg: leg_at(i),
                        attempt: u32::try_from(attempt).unwrap_or(u32::MAX),
                    });
                }
            }
        }
        actions
    }
}

fn exposure(
    near: Decimal,
    near_size: Decimal,
    far: Decimal,
    far_size: Decimal,
) -> Result<Decimal, RollError> {
    exact_add(exact_mul(near, near_size)?, exact_mul(far, far_size)?)
}

#[cfg(feature = "margin")]
mod ledger {
    use super::*;
    use margin_rs::ledger::Ledger;

    /// What a ledger holds about a leg of a roll.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct RollLegReconciliation {
        /// The position the plan expects.
        pub expected_position: Decimal,
        /// The ledger's position.
        pub ledger_position: Decimal,
        /// `ledger_position - expected_position`: fills the plan does not know, such as
        /// a quote's, or a fill the ledger does not hold.
        pub unexplained: Decimal,
        /// The realized PnL of the plan's own fills that the ledger holds.
        pub roll_realized: Decimal,
        /// The fees of the plan's own fills that it holds, per asset.
        pub roll_fees: BTreeMap<String, Decimal>,
        /// The whole position's realized PnL, with any history before the roll.
        pub position_realized: Decimal,
        /// The ledger's executions on this contract since the anchor that are not the
        /// plan's, even if they leave the position as it was.
        pub unexplained_entries: u64,
        /// Their realized PnL.
        pub unexplained_realized: Decimal,
        /// Their fees, per asset.
        pub unexplained_fees: BTreeMap<String, Decimal>,
    }

    /// The near contract's settlement in a ledger, apart from the fills' cash.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct RollSettlement {
        /// The contracts closed, signed like the position.
        pub contracts: Decimal,
        /// The settlement price.
        pub price: Decimal,
        /// Its realized PnL.
        pub realized_pnl: Decimal,
    }

    /// A difference between a plan and a ledger.
    #[derive(Debug, Clone, PartialEq, Eq)]
    #[non_exhaustive]
    pub enum RollDiscrepancy {
        /// A fill of the plan that the ledger does not hold, such as one it refused at
        /// the expiry.
        MissingInLedger {
            /// The leg.
            leg: RollLeg,
            /// The execution.
            exec_id: String,
        },
        /// A fill the ledger holds with other terms.
        FillDiffers {
            /// The leg.
            leg: RollLeg,
            /// The execution.
            exec_id: String,
            /// What differs: `side`, `price`, `contracts`, `fee`, `fee_asset` or
            /// `closed_at_settlement`.
            field: &'static str,
        },
        /// Executions since the anchor that are not the plan's: their cash is in
        /// [`RollLegReconciliation`].
        UnexplainedEntries {
            /// The leg.
            leg: RollLeg,
            /// How many.
            count: u64,
        },
        /// A position the plan does not explain.
        UnexplainedPosition {
            /// The leg.
            leg: RollLeg,
            /// `ledger_position - expected_position`.
            quantity: Decimal,
        },
        /// The ledger settled the near contract, and the plan was not told.
        PlanUnawareOfSettlement,
        /// The plan was told of a settlement that the ledger does not hold.
        LedgerNotSettled,
        /// The plan and the ledger settled different contracts.
        SettledContractsDiffer {
            /// What the plan recorded.
            plan: Decimal,
            /// What the ledger closed.
            ledger: Decimal,
        },
        /// The plan and the ledger settled at different prices.
        SettledPriceDiffers {
            /// What the plan recorded.
            plan: Decimal,
            /// What the ledger booked.
            ledger: Decimal,
        },
    }

    /// The result of [`RollPlan::reconcile`].
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct RollReconciliation {
        /// The near contract.
        pub near: RollLegReconciliation,
        /// The far contract.
        pub far: RollLegReconciliation,
        /// The near contract's settlement, if the ledger holds one.
        pub settlement: Option<RollSettlement>,
        /// What differs: empty if the plan and the ledger agree.
        pub discrepancies: Vec<RollDiscrepancy>,
    }

    impl RollReconciliation {
        /// Returns `true` if the plan and the ledger agree: on positions, on the plan's
        /// fills and on every execution since the anchor.
        #[must_use]
        pub fn is_clean(&self) -> bool {
            self.discrepancies.is_empty()
        }
    }

    /// The journal entry of a fill: under the identifier
    /// [`fill_execution`](crate::instruments::margin::fill_execution) gives, or under its
    /// `exec_id` alone, as `backtest::contracts::ContractEngine` books them, and only if
    /// that entry is on this contract.
    fn entry_of<'a>(
        ledger: &'a Ledger,
        id: &margin_rs::InstrumentId,
        instrument: &str,
        exec_id: &str,
    ) -> Option<&'a margin_rs::ledger::ExecutionCashflow> {
        let named = crate::instruments::margin::execution_id(instrument, exec_id)
            .ok()
            .and_then(|execution| ledger.execution_cashflow(&execution));
        named
            .or_else(|| {
                margin_rs::ExecutionId::new(exec_id)
                    .ok()
                    .and_then(|execution| ledger.execution_cashflow(&execution))
            })
            .filter(|entry| entry.instrument == *id)
    }

    impl RollPlan {
        /// Anchors the plan to `ledger` as it stands now, so that
        /// [`RollPlan::reconcile`] can tell what happened since from its history.
        #[must_use]
        pub fn anchored_to(mut self, ledger: &Ledger) -> Self {
            self.journal_anchor = Some(u64::try_from(ledger.journal().len()).unwrap_or(u64::MAX));
            self
        }

        /// Compares this plan with `ledger`, which it does not change.
        ///
        /// A fill's identifier in the ledger is the one
        /// [`fill_execution`](crate::instruments::margin::fill_execution) gives, from
        /// its `exec_id` as the trade identifier, or its `exec_id` alone on this contract,
        /// as `backtest::contracts::ContractEngine` books them. The journal keeps no
        /// execution time, so that is not compared. In a run, the ledger books a fill
        /// when it matches and the plan hears of it later, so this reads clean only once
        /// the run ended and every report was delivered.
        ///
        /// # Errors
        ///
        /// [`RollError::NotAnchored`] without [`RollPlan::anchored_to`];
        /// [`RollError::Future`] if the ledger holds other terms under an identifier;
        /// and [`RollError::Plan`] if a sum overflows.
        pub fn reconcile(&self, ledger: &Ledger) -> Result<RollReconciliation, RollError> {
            if self.journal_anchor.is_none() {
                return Err(RollError::NotAnchored);
            }
            let mut discrepancies = Vec::new();
            let near_id = self.near.terms.identify_in_ledger(ledger)?;
            let settlement = ledger
                .expiry_journal()
                .iter()
                .find(|e| e.instrument == near_id)
                .map(|e| RollSettlement {
                    contracts: e.contracts,
                    price: e.price,
                    realized_pnl: e.realized_pnl,
                });
            match (&self.settled, settlement) {
                (None, Some(_)) => discrepancies.push(RollDiscrepancy::PlanUnawareOfSettlement),
                (Some(_), None) => discrepancies.push(RollDiscrepancy::LedgerNotSettled),
                (Some(plan), Some(ledger)) => {
                    if plan.contracts != ledger.contracts {
                        discrepancies.push(RollDiscrepancy::SettledContractsDiffer {
                            plan: plan.contracts,
                            ledger: ledger.contracts,
                        });
                    }
                    if plan.price.price() != ledger.price {
                        discrepancies.push(RollDiscrepancy::SettledPriceDiffers {
                            plan: plan.price.price(),
                            ledger: ledger.price,
                        });
                    }
                }
                (None, None) => {}
            }
            // What the plan was not told still left the ledger's position.
            let unknown = match (&self.settled, settlement) {
                (None, Some(ledger)) => ledger.contracts,
                _ => Decimal::ZERO,
            };
            let near = self.reconcile_leg(RollLeg::Near, ledger, unknown, &mut discrepancies)?;
            let far =
                self.reconcile_leg(RollLeg::Far, ledger, Decimal::ZERO, &mut discrepancies)?;
            Ok(RollReconciliation {
                near,
                far,
                settlement,
                discrepancies,
            })
        }

        fn reconcile_leg(
            &self,
            leg: RollLeg,
            ledger: &Ledger,
            settled_unknown: Decimal,
            discrepancies: &mut Vec<RollDiscrepancy>,
        ) -> Result<RollLegReconciliation, RollError> {
            let terms = &self.leg(leg).terms;
            let id = terms.identify_in_ledger(ledger)?;
            let instrument = terms.spec().id().as_str();
            let mut roll_realized = Decimal::ZERO;
            let mut roll_fees = BTreeMap::new();
            for fill in self.fills.iter().filter(|f| f.leg == leg) {
                let Some(entry) = entry_of(ledger, &id, instrument, &fill.exec_id) else {
                    discrepancies.push(RollDiscrepancy::MissingInLedger {
                        leg,
                        exec_id: fill.exec_id.clone(),
                    });
                    continue;
                };
                let (fee, fee_asset) = entry.fee.as_ref().map_or((Decimal::ZERO, None), |f| {
                    (f.amount, Some(f.asset.as_str()))
                });
                let mut differs = |field| {
                    discrepancies.push(RollDiscrepancy::FillDiffers {
                        leg,
                        exec_id: fill.exec_id.clone(),
                        field,
                    });
                };
                if entry.side != fill.side {
                    differs("side");
                }
                if entry.price != fill.price {
                    differs("price");
                }
                if entry.quantity != fill.contracts {
                    differs("contracts");
                }
                if fee != fill.fee {
                    differs("fee");
                }
                if !fill.fee.is_zero() && fee_asset != Some(fill.fee_asset.as_str()) {
                    differs("fee_asset");
                }
                // The ledger closes such a fill whole. What the plan told apart must be the
                // same quantity, which a flag alone would not show.
                let closed_in_ledger = entry
                    .derivative
                    .as_ref()
                    .is_some_and(|d| d.closed_at_settlement.is_some());
                let closed_in_plan = self
                    .closed_fills
                    .iter()
                    .find(|(id, _)| leg == RollLeg::Near && *id == fill.exec_id)
                    .map_or(Decimal::ZERO, |(_, closed)| *closed);
                let whole = if entry.side == Action::Buy {
                    entry.quantity
                } else {
                    arbitrage_rs::exact::neg(entry.quantity)
                };
                let expected = if closed_in_ledger {
                    whole
                } else {
                    Decimal::ZERO
                };
                if closed_in_plan != expected {
                    differs("closed_at_settlement");
                }
                roll_realized = exact_add(roll_realized, entry.realized_pnl)?;
                if let Some(paid) = &entry.fee {
                    let sum = roll_fees
                        .entry(paid.asset.as_str().to_string())
                        .or_insert(Decimal::ZERO);
                    *sum = exact_add(*sum, paid.amount)?;
                }
            }
            // Since the anchor, what is on this contract and is not the plan's.
            let anchor = self.journal_anchor.unwrap_or(u64::MAX);
            let mut plan_ids = Vec::new();
            for fill in self.fills.iter().filter(|f| f.leg == leg) {
                if let Some(entry) = entry_of(ledger, &id, instrument, &fill.exec_id) {
                    plan_ids.push(entry.execution_id.clone());
                }
            }
            let mut unexplained_entries = 0_u64;
            let mut unexplained_realized = Decimal::ZERO;
            let mut unexplained_fees = BTreeMap::new();
            for entry in ledger.journal().iter().filter(|e| {
                e.instrument == id && e.sequence >= anchor && !plan_ids.contains(&e.execution_id)
            }) {
                unexplained_entries = unexplained_entries.saturating_add(1);
                unexplained_realized = exact_add(unexplained_realized, entry.realized_pnl)?;
                if let Some(paid) = &entry.fee {
                    let sum = unexplained_fees
                        .entry(paid.asset.as_str().to_string())
                        .or_insert(Decimal::ZERO);
                    *sum = exact_add(*sum, paid.amount)?;
                }
            }
            if unexplained_entries > 0 {
                discrepancies.push(RollDiscrepancy::UnexplainedEntries {
                    leg,
                    count: unexplained_entries,
                });
            }
            let position = ledger.position(&id);
            let ledger_position = position.as_ref().map_or(Decimal::ZERO, |p| p.quantity);
            let expected_position = exact_sub(self.position(leg), settled_unknown)?;
            let unexplained = exact_sub(ledger_position, expected_position)?;
            if !unexplained.is_zero() {
                discrepancies.push(RollDiscrepancy::UnexplainedPosition {
                    leg,
                    quantity: unexplained,
                });
            }
            Ok(RollLegReconciliation {
                expected_position,
                ledger_position,
                unexplained,
                roll_realized,
                roll_fees,
                position_realized: position.as_ref().map_or(Decimal::ZERO, |p| p.realized_pnl),
                unexplained_entries,
                unexplained_realized,
                unexplained_fees,
            })
        }
    }
}

#[cfg(feature = "margin")]
pub use ledger::{RollDiscrepancy, RollLegReconciliation, RollReconciliation, RollSettlement};
