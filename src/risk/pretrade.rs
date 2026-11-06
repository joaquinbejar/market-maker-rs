//! Pretrade risk port.
//!
//! Every order passes a [`PretradePolicy`](crate::risk::pretrade::PretradePolicy) before it is submitted. The
//! policy accepts it, reduces its quantity or rejects it, and is told about
//! the order's lifecycle afterwards (accepts, fills, cancels) so it can track
//! live exposure and reservations. The same seam serves the quote path of
//! the backtest engine
//! ([`BacktestEngine::with_pretrade_policy`](crate::backtest::BacktestEngine::with_pretrade_policy))
//! and the live hedge or quote path of
//! [`OrderManager::register_order_checked`](crate::execution::OrderManager::register_order_checked).
//!
//! [`LimitsPolicy`](crate::risk::pretrade::LimitsPolicy) is a local policy built on [`RiskLimits`] and
//! [`CircuitBreaker`]. With the `margin` feature, the backtest module offers
//! an adapter over the `margin-rs` risk gate.
//!
//! # Example
//!
//! ```rust
//! use market_maker_rs::dec;
//! use market_maker_rs::execution::Action;
//! use market_maker_rs::risk::RiskLimits;
//! use market_maker_rs::risk::pretrade::{
//!     LimitsPolicy, OrderPurpose, PretradeDecision, PretradePolicy, PretradeRequest,
//! };
//!
//! let limits = RiskLimits::new(dec!(10), dec!(1_000_000), dec!(0)).unwrap();
//! let mut policy = LimitsPolicy::new(limits);
//! let request = PretradeRequest::new("o-1", Action::Buy, Some(dec!(100)), dec!(15), OrderPurpose::Quote, 0);
//! let decision = policy.check(&request);
//! assert_eq!(decision.approved_quantity(request.quantity), dec!(10));
//! assert!(matches!(decision, PretradeDecision::Reduced { .. }));
//! ```

use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::Decimal;
use crate::execution::Action;
use crate::types::action::Dir;
use crate::types::error::{MMError, MMResult};
use crate::types::fee::FeeAsset;

use super::{CircuitBreaker, RiskLimits};

/// Why an order is sent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OrderPurpose {
    /// Market making quote.
    Quote,
    /// Hedge of inventory.
    Hedge,
    /// Anything else.
    Other,
}

/// Order presented to a [`PretradePolicy`] before submission.
#[derive(Debug, Clone, PartialEq)]
pub struct PretradeRequest {
    /// Order identifier, unique per policy.
    pub order_id: String,
    /// Instrument the order trades, when the caller knows it. Policies bound
    /// to one instrument reject a request for another.
    pub instrument: Option<String>,
    /// Order side. Only buy and sell are supported; policies reject any
    /// other action.
    pub side: Action,
    /// Limit price; `None` for a market order.
    pub price: Option<Decimal>,
    /// Requested quantity.
    pub quantity: Decimal,
    /// Why the order is sent.
    pub purpose: OrderPurpose,
    /// Decision time, in milliseconds.
    pub timestamp: u64,
}

impl PretradeRequest {
    /// Creates a request.
    #[must_use]
    pub fn new(
        order_id: impl Into<String>,
        side: Action,
        price: Option<Decimal>,
        quantity: Decimal,
        purpose: OrderPurpose,
        timestamp: u64,
    ) -> Self {
        Self {
            order_id: order_id.into(),
            instrument: None,
            side,
            price,
            quantity,
            purpose,
            timestamp,
        }
    }

    /// Names the instrument the order trades.
    #[must_use]
    pub fn with_instrument(mut self, instrument: impl Into<String>) -> Self {
        self.instrument = Some(instrument.into());
        self
    }
}

/// Decision of a [`PretradePolicy`].
#[derive(Debug, Clone, PartialEq)]
pub enum PretradeDecision {
    /// Submit the order as requested.
    Accepted,
    /// Submit a smaller quantity.
    Reduced {
        /// Approved quantity.
        quantity: Decimal,
        /// Why the order was reduced.
        reason: String,
    },
    /// Do not submit.
    Rejected {
        /// Why the order was rejected.
        reason: String,
    },
}

impl PretradeDecision {
    /// Quantity that may be submitted for a request of `requested`.
    ///
    /// A reduction can never increase the quantity: the result is clamped to
    /// `0..=requested`. A rejection approves zero.
    #[must_use]
    pub fn approved_quantity(&self, requested: Decimal) -> Decimal {
        match self {
            Self::Accepted => requested,
            Self::Reduced { quantity, .. } => (*quantity).min(requested).max(Decimal::ZERO),
            Self::Rejected { .. } => Decimal::ZERO,
        }
    }

    /// Returns true for a rejection.
    #[must_use]
    pub fn is_rejected(&self) -> bool {
        matches!(self, Self::Rejected { .. })
    }
}

/// Lifecycle event of an order the policy approved.
///
/// Events may also arrive for orders the policy never saw (for example an
/// order rejected locally before the policy was asked); policies ignore
/// unknown ids.
#[derive(Debug, Clone, PartialEq)]
pub enum OrderEvent {
    /// The venue accepted the order.
    Accepted {
        /// Order identifier.
        order_id: String,
    },
    /// The venue rejected the order.
    VenueRejected {
        /// Order identifier.
        order_id: String,
    },
    /// A cancel was sent; the order stays live until it is confirmed.
    CancelRequested {
        /// Order identifier.
        order_id: String,
    },
    /// The venue confirmed the cancel; the remainder is gone.
    Canceled {
        /// Order identifier.
        order_id: String,
    },
    /// The cancel arrived too late; the order had already ended.
    CancelRejected {
        /// Order identifier.
        order_id: String,
    },
    /// An execution.
    Fill {
        /// Order identifier.
        order_id: String,
        /// Unique execution identifier.
        execution_id: String,
        /// Order side. Only buy and sell are supported: a fill with any other
        /// action is unknown exposure, so the built-in policies return
        /// [`MMError::UnsupportedAction`]
        /// for it and reject every later order until they are reset.
        side: Action,
        /// Execution price.
        price: Decimal,
        /// Quantity of this execution.
        quantity: Decimal,
        /// Cumulative filled quantity of the order after this execution.
        cumulative_quantity: Decimal,
        /// Fee amount, in [`OrderEvent::Fill::fee_asset`] units.
        fee: Decimal,
        /// Asset in which the fee is charged.
        fee_asset: FeeAsset,
        /// Venue time, in milliseconds.
        timestamp: u64,
    },
}

/// Pretrade risk policy.
///
/// # Contract
///
/// - [`PretradePolicy::check`] is called once per order, before it is
///   submitted; callers never submit more than
///   [`PretradeDecision::approved_quantity`].
/// - Callers forward the order's lifecycle with
///   [`PretradePolicy::on_order_event`], in venue order. An error means the
///   policy could not account for the event: callers stop submitting new
///   orders (the built-in policies also reject them from then on).
/// - [`PretradePolicy::reset`] returns the policy to its initial state.
pub trait PretradePolicy: fmt::Debug {
    /// Decides on an order before submission.
    ///
    /// A request whose side is neither buy nor sell is rejected before any
    /// state changes.
    fn check(&mut self, request: &PretradeRequest) -> PretradeDecision;

    /// Market mark observed at `timestamp`.
    fn on_market_data(&mut self, _mark: Decimal, _timestamp: u64) {}

    /// Lifecycle event of an order.
    ///
    /// # Errors
    ///
    /// The built-in policies return
    /// [`MMError::UnsupportedAction`]
    /// for a fill whose side is neither buy nor sell, and from then on
    /// reject every [`PretradePolicy::check`] until
    /// [`PretradePolicy::reset`]. The default implementation ignores every
    /// event.
    fn on_order_event(&mut self, _event: &OrderEvent) -> MMResult<()> {
        Ok(())
    }

    /// Returns to the initial state.
    fn reset(&mut self);

    /// Human readable description.
    fn description(&self) -> String {
        "custom pretrade policy".to_string()
    }

    /// Whether it decides on quantities in contracts, on requests that name their
    /// contract, without reading `price * quantity` as a notional, across contracts,
    /// with a [`PretradePolicy::on_market_data`] mark that may be of any of them.
    /// `backtest::contracts::ContractEngine` refuses a policy that does not declare it.
    fn in_contracts(&self) -> bool {
        false
    }

    /// Whether it decides in contracts for the one contract of a run, reading each
    /// [`PretradePolicy::on_market_data`] mark as that contract's. Less than
    /// [`PretradePolicy::in_contracts`], which also copes with marks of other contracts.
    /// `BacktestEngine::with_instrument` refuses a policy that declares neither.
    fn in_one_contract(&self) -> bool {
        self.in_contracts()
    }
}

/// Outcome of [`limits_capacity`] for one order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CapacityVerdict {
    /// The whole requested quantity fits.
    Accept,
    /// Only this smaller, positive quantity fits.
    Reduce(Decimal),
    /// Nothing fits.
    Reject(RejectCode),
}

/// Why [`limits_capacity`] rejected an order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RejectCode {
    /// A position or notional limit leaves no room.
    LimitReached,
}

/// Room left before one limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LimitCap {
    /// The limit leaves this much room (possibly negative).
    Binding(Decimal),
    /// The room is not representable: the limit cannot bind.
    NonBinding,
}

/// The worst case exposure `position + live` is not representable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GateNumericError;

/// Capacity of a new order under `limits`, as [`LimitsPolicy::check`]
/// decides it.
///
/// `position` is the filled position, `live_same_side` the remaining
/// quantity of the live orders on the order's side, `price` the order price
/// (positive) and `qty` the requested quantity (positive). A limit whose
/// room overflows, or a notional limit whose `max_notional / price` is not
/// representable, is [`LimitCap::NonBinding`].
///
/// # Errors
///
/// Returns [`GateNumericError`] if `position + live` (signed by side) is
/// not representable.
pub(crate) fn limits_capacity(
    limits: &RiskLimits,
    position: Decimal,
    live_same_side: Decimal,
    dir: Dir,
    price: Decimal,
    qty: Decimal,
) -> Result<CapacityVerdict, GateNumericError> {
    let worst = position
        .checked_add(LimitsPolicy::signed(dir, live_same_side))
        .ok_or(GateNumericError)?;
    let room = |limit: Option<Decimal>| match limit {
        None => LimitCap::NonBinding,
        Some(limit) => {
            let room = match dir {
                Dir::Buy => limit.checked_sub(worst),
                Dir::Sell => limit.checked_add(worst),
            };
            room.map_or(LimitCap::NonBinding, LimitCap::Binding)
        }
    };
    let mut capacity = qty;
    for cap in [
        room(Some(limits.max_position)),
        room(limits.max_notional.checked_div(price)),
    ] {
        match cap {
            LimitCap::Binding(room) => capacity = capacity.min(room),
            LimitCap::NonBinding => {}
        }
    }
    Ok(if capacity >= qty {
        CapacityVerdict::Accept
    } else if capacity > Decimal::ZERO {
        CapacityVerdict::Reduce(capacity)
    } else {
        CapacityVerdict::Reject(RejectCode::LimitReached)
    })
}

/// Local [`PretradePolicy`] over [`RiskLimits`] and an optional
/// [`CircuitBreaker`].
///
/// The position limit applies to the worst case: current position plus the
/// remaining quantity of every live order on the same side plus the new
/// order. An order that would exceed it is reduced to the remaining
/// capacity, or rejected when none is left. The notional limit uses the
/// order price (or the last mark for a market order). While the breaker does
/// not allow trading, every order is rejected.
///
/// A fill whose side is neither buy nor sell latches a fault
/// ([`LimitsPolicy::fault`]): the position is then unknown, so every order
/// is rejected until [`PretradePolicy::reset`].
#[derive(Debug, Clone)]
pub struct LimitsPolicy {
    limits: RiskLimits,
    instrument: Option<String>,
    initial_breaker: Option<CircuitBreaker>,
    breaker: Option<CircuitBreaker>,
    executions: HashSet<String>,
    position: Decimal,
    mark: Option<Decimal>,
    live: HashMap<String, (Dir, Decimal)>,
    fault: Option<MMError>,
}

impl LimitsPolicy {
    /// Creates a policy without a circuit breaker.
    #[must_use]
    pub fn new(limits: RiskLimits) -> Self {
        Self {
            limits,
            instrument: None,
            initial_breaker: None,
            breaker: None,
            executions: HashSet::new(),
            position: Decimal::ZERO,
            mark: None,
            live: HashMap::new(),
            fault: None,
        }
    }

    /// Adds a circuit breaker whose state gates every order.
    #[must_use]
    pub fn with_circuit_breaker(mut self, breaker: CircuitBreaker) -> Self {
        self.initial_breaker = Some(breaker.clone());
        self.breaker = Some(breaker);
        self
    }

    /// Binds the policy to one instrument: requests naming another are
    /// rejected.
    #[must_use]
    pub fn with_instrument(mut self, instrument: impl Into<String>) -> Self {
        self.instrument = Some(instrument.into());
        self
    }

    /// The circuit breaker, to trigger or update it.
    pub fn circuit_breaker_mut(&mut self) -> Option<&mut CircuitBreaker> {
        self.breaker.as_mut()
    }

    /// The fault latched by a fill with an unsupported side, if any. While
    /// it is set, every order is rejected; [`PretradePolicy::reset`] clears
    /// it.
    #[must_use]
    pub fn fault(&self) -> Option<&MMError> {
        self.fault.as_ref()
    }

    /// Position implied by the fills seen so far.
    #[must_use]
    pub fn position(&self) -> Decimal {
        self.position
    }

    /// Remaining quantity of live orders on `side`.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::UnsupportedAction`]
    /// if `side` is neither buy nor sell, or [`MMError::NumericalError`]
    /// if the live quantity overflows.
    pub fn live_quantity(&self, side: Action) -> MMResult<Decimal> {
        let dir = Dir::try_from(side)?;
        self.live_quantity_dir(dir)
            .ok_or_else(|| MMError::NumericalError(format!("live {side:?} quantity overflows")))
    }

    /// Sum of the live quantities on `dir`; `None` on overflow.
    fn live_quantity_dir(&self, dir: Dir) -> Option<Decimal> {
        self.live
            .values()
            .filter(|(d, _)| *d == dir)
            .try_fold(Decimal::ZERO, |acc, (_, q)| acc.checked_add(*q))
    }

    fn signed(dir: Dir, quantity: Decimal) -> Decimal {
        match dir {
            Dir::Buy => quantity,
            // Negating a Decimal flips its sign bit: it cannot overflow.
            #[allow(clippy::arithmetic_side_effects)]
            Dir::Sell => -quantity,
        }
    }
}

impl PretradePolicy for LimitsPolicy {
    fn check(&mut self, request: &PretradeRequest) -> PretradeDecision {
        // Before the breaker cooldown below, the first state change.
        let dir = match Dir::try_from(request.side) {
            Ok(dir) => dir,
            Err(error) => {
                return PretradeDecision::Rejected {
                    reason: error.to_string(),
                };
            }
        };
        if let Some(fault) = &self.fault {
            return PretradeDecision::Rejected {
                reason: format!("policy fault: {fault}"),
            };
        }
        if let Some(breaker) = self.breaker.as_mut() {
            let _ = breaker.check_cooldown(request.timestamp);
            if !breaker.is_trading_allowed() {
                return PretradeDecision::Rejected {
                    reason: "circuit breaker does not allow trading".to_string(),
                };
            }
        }
        if request.quantity <= Decimal::ZERO {
            return PretradeDecision::Rejected {
                reason: "quantity must be positive".to_string(),
            };
        }
        if let (Some(bound), Some(asked)) = (&self.instrument, &request.instrument)
            && bound != asked
        {
            return PretradeDecision::Rejected {
                reason: format!("policy is bound to {bound}, not {asked}"),
            };
        }
        if self.live.contains_key(&request.order_id) {
            return PretradeDecision::Rejected {
                reason: format!("duplicate order id {}", request.order_id),
            };
        }
        let Some(price) = request.price.or(self.mark) else {
            return PretradeDecision::Rejected {
                reason: "no price and no market mark".to_string(),
            };
        };
        if price <= Decimal::ZERO {
            return PretradeDecision::Rejected {
                reason: "price must be positive".to_string(),
            };
        }

        // Worst case exposure on the order's side before this order. An
        // exposure that cannot be represented is rejected, never approved.
        let verdict = self
            .live_quantity_dir(dir)
            .ok_or(GateNumericError)
            .and_then(|live| {
                limits_capacity(
                    &self.limits,
                    self.position,
                    live,
                    dir,
                    price,
                    request.quantity,
                )
            });
        let decision = match verdict {
            Ok(CapacityVerdict::Accept) => PretradeDecision::Accepted,
            Ok(CapacityVerdict::Reduce(quantity)) => PretradeDecision::Reduced {
                quantity,
                reason: "position or notional limit".to_string(),
            },
            Ok(CapacityVerdict::Reject(RejectCode::LimitReached)) => PretradeDecision::Rejected {
                reason: "position or notional limit reached".to_string(),
            },
            Err(GateNumericError) => {
                return PretradeDecision::Rejected {
                    reason: "exposure overflows decimal range".to_string(),
                };
            }
        };
        let approved = decision.approved_quantity(request.quantity);
        if approved > Decimal::ZERO {
            self.live.insert(request.order_id.clone(), (dir, approved));
        }
        decision
    }

    fn on_market_data(&mut self, mark: Decimal, _timestamp: u64) {
        if mark > Decimal::ZERO {
            self.mark = Some(mark);
        }
    }

    fn on_order_event(&mut self, event: &OrderEvent) -> MMResult<()> {
        match event {
            OrderEvent::Accepted { .. }
            | OrderEvent::CancelRequested { .. }
            | OrderEvent::CancelRejected { .. } => {}
            OrderEvent::VenueRejected { order_id } | OrderEvent::Canceled { order_id } => {
                self.live.remove(order_id);
            }
            OrderEvent::Fill {
                order_id,
                execution_id,
                side,
                quantity,
                fee,
                fee_asset,
                ..
            } => {
                // A fill with an unsupported side is unknown exposure: latch
                // the fault and fail. Nothing else changes, not even the
                // replay set, so a corrected redelivery still applies.
                let dir = match Dir::try_from(*side) {
                    Ok(dir) => dir,
                    Err(error) => {
                        self.fault = Some(error.clone());
                        return Err(error);
                    }
                };
                // A replayed execution changes nothing.
                if self.executions.contains(execution_id) {
                    return Ok(());
                }
                // A fee paid in base reduces the base held, like the
                // engine's position: a buy receives `quantity - fee`, a
                // sell delivers `quantity + fee`.
                let base_fee = match fee_asset {
                    FeeAsset::Base => *fee,
                    FeeAsset::Quote | FeeAsset::Other(_) | FeeAsset::Settlement => Decimal::ZERO,
                };
                // The position is a running balance: an overflow is unknown
                // exposure, so it latches the fault like an unsupported
                // side and changes nothing.
                let position = Self::signed(dir, *quantity)
                    .checked_sub(base_fee)
                    .and_then(|delta| self.position.checked_add(delta));
                let Some(position) = position else {
                    let error = MMError::NumericalError(format!(
                        "position overflows: {} after a fill of {} on execution {}",
                        self.position, quantity, execution_id
                    ));
                    self.fault = Some(error.clone());
                    return Err(error);
                };
                self.executions.insert(execution_id.clone());
                self.position = position;
                if let Some((_, remaining)) = self.live.get_mut(order_id) {
                    *remaining = remaining.saturating_sub(*quantity);
                    if *remaining <= Decimal::ZERO {
                        self.live.remove(order_id);
                    }
                }
            }
        }
        Ok(())
    }

    fn reset(&mut self) {
        self.fault = None;
        self.position = Decimal::ZERO;
        self.mark = None;
        self.live.clear();
        self.executions.clear();
        self.breaker = self.initial_breaker.clone();
    }

    fn description(&self) -> String {
        format!(
            "local limits: max position {}, max notional {}{}",
            self.limits.max_position,
            self.limits.max_notional,
            if self.breaker.is_some() {
                ", circuit breaker"
            } else {
                ""
            }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dec;
    use crate::risk::CircuitBreakerConfig;

    fn policy(max_position: Decimal) -> LimitsPolicy {
        LimitsPolicy::new(RiskLimits::new(max_position, dec!(1_000_000), dec!(0)).unwrap())
    }

    fn buy(id: &str, quantity: Decimal) -> PretradeRequest {
        PretradeRequest::new(
            id,
            Action::Buy,
            Some(dec!(100)),
            quantity,
            OrderPurpose::Quote,
            0,
        )
    }

    fn unbounded_short_policy() -> LimitsPolicy {
        let limits = RiskLimits::new(Decimal::MAX, Decimal::MAX, dec!(0)).unwrap();
        let mut policy = LimitsPolicy::new(limits);
        policy
            .on_order_event(&OrderEvent::Fill {
                order_id: "short".to_string(),
                execution_id: "e0".to_string(),
                side: Action::Sell,
                price: dec!(1),
                quantity: Decimal::MAX,
                cumulative_quantity: Decimal::MAX,
                fee: Decimal::ZERO,
                fee_asset: crate::types::fee::FeeAsset::Quote,
                timestamp: 0,
            })
            .unwrap();
        policy
    }

    fn buy_at_one(id: &str, quantity: Decimal) -> PretradeRequest {
        PretradeRequest::new(
            id,
            Action::Buy,
            Some(dec!(1)),
            quantity,
            OrderPurpose::Quote,
            0,
        )
    }

    #[test]
    fn test_fill_position_overflow_latches_fault() {
        let mut policy = unbounded_short_policy();
        let fill = OrderEvent::Fill {
            order_id: "more".to_string(),
            execution_id: "e1".to_string(),
            side: Action::Sell,
            price: dec!(1),
            quantity: dec!(1),
            cumulative_quantity: dec!(1),
            fee: Decimal::ZERO,
            fee_asset: crate::types::fee::FeeAsset::Quote,
            timestamp: 0,
        };
        let result = policy.on_order_event(&fill);
        assert!(
            matches!(result, Err(MMError::NumericalError(_))),
            "{result:?}"
        );
        assert_eq!(policy.position(), -Decimal::MAX);
        assert!(matches!(
            policy.check(&buy_at_one("a", dec!(1))),
            PretradeDecision::Rejected { .. }
        ));
    }

    #[test]
    fn test_check_room_beyond_decimal_range_accepted() {
        let mut policy = unbounded_short_policy();
        assert_eq!(policy.position(), -Decimal::MAX);
        let decision = policy.check(&buy_at_one("a", Decimal::MAX));
        assert_eq!(decision, PretradeDecision::Accepted);
    }

    #[test]
    fn test_check_live_exposure_overflow_rejected() {
        let mut policy = unbounded_short_policy();
        assert_eq!(
            policy.check(&buy_at_one("a", Decimal::MAX)),
            PretradeDecision::Accepted
        );
        assert_eq!(
            policy.check(&buy_at_one("b", Decimal::MAX)),
            PretradeDecision::Accepted
        );
        let decision = policy.check(&buy_at_one("c", dec!(1)));
        assert!(matches!(decision, PretradeDecision::Rejected { .. }));
        assert!(matches!(
            policy.live_quantity(Action::Buy),
            Err(MMError::NumericalError(_))
        ));
    }

    #[test]
    fn test_reduction_never_increases_quantity() {
        let reduced = PretradeDecision::Reduced {
            quantity: dec!(5),
            reason: String::new(),
        };
        assert_eq!(reduced.approved_quantity(dec!(3)), dec!(3));
        assert_eq!(reduced.approved_quantity(dec!(8)), dec!(5));
        let negative = PretradeDecision::Reduced {
            quantity: dec!(-1),
            reason: String::new(),
        };
        assert_eq!(negative.approved_quantity(dec!(3)), Decimal::ZERO);
        let rejected = PretradeDecision::Rejected {
            reason: String::new(),
        };
        assert_eq!(rejected.approved_quantity(dec!(3)), Decimal::ZERO);
    }

    #[test]
    fn test_live_orders_count_towards_the_limit() {
        let mut policy = policy(dec!(10));
        assert_eq!(policy.check(&buy("a", dec!(6))), PretradeDecision::Accepted);
        let second = policy.check(&buy("b", dec!(6)));
        assert_eq!(second.approved_quantity(dec!(6)), dec!(4));
        assert!(policy.check(&buy("c", dec!(1))).is_rejected());
        // Cancel frees capacity; a fill keeps it used.
        policy
            .on_order_event(&OrderEvent::Canceled {
                order_id: "b".to_string(),
            })
            .unwrap();
        policy
            .on_order_event(&OrderEvent::Fill {
                order_id: "a".to_string(),
                execution_id: "e1".to_string(),
                side: Action::Buy,
                price: dec!(100),
                quantity: dec!(6),
                cumulative_quantity: dec!(6),
                fee: Decimal::ZERO,
                fee_asset: crate::types::fee::FeeAsset::Quote,
                timestamp: 1,
            })
            .unwrap();
        assert_eq!(policy.position(), dec!(6));
        assert_eq!(policy.live_quantity(Action::Buy).unwrap(), Decimal::ZERO);
        assert_eq!(
            policy.check(&buy("d", dec!(9))).approved_quantity(dec!(9)),
            dec!(4)
        );
    }

    #[test]
    fn test_sell_side_has_its_own_capacity() {
        let mut policy = policy(dec!(10));
        let sell = PretradeRequest::new(
            "s",
            Action::Sell,
            Some(dec!(100)),
            dec!(10),
            OrderPurpose::Hedge,
            0,
        );
        assert_eq!(policy.check(&sell), PretradeDecision::Accepted);
    }

    #[test]
    fn test_market_order_uses_mark_or_is_rejected() {
        let mut policy = policy(dec!(10));
        let market = PretradeRequest::new("m", Action::Buy, None, dec!(1), OrderPurpose::Hedge, 0);
        assert!(policy.check(&market).is_rejected());
        policy.on_market_data(dec!(100), 0);
        assert_eq!(policy.check(&market), PretradeDecision::Accepted);
    }

    #[test]
    fn test_circuit_breaker_rejects_everything() {
        let config =
            CircuitBreakerConfig::new(dec!(1000), dec!(0.05), 5, dec!(0.10), 300_000, 60_000)
                .unwrap();
        let mut policy = policy(dec!(10)).with_circuit_breaker(CircuitBreaker::new(config));
        policy.circuit_breaker_mut().unwrap().trigger_manual(0);
        assert!(policy.check(&buy("a", dec!(1))).is_rejected());
    }

    #[test]
    fn test_tiny_price_does_not_panic() {
        let mut policy = policy(dec!(10));
        let tiny = PretradeRequest::new(
            "t",
            Action::Buy,
            Some(Decimal::new(1, 28)),
            dec!(1),
            OrderPurpose::Quote,
            0,
        );
        assert_eq!(policy.check(&tiny), PretradeDecision::Accepted);
    }

    #[test]
    fn test_replayed_fill_is_ignored() {
        let mut policy = policy(dec!(10));
        assert_eq!(policy.check(&buy("a", dec!(5))), PretradeDecision::Accepted);
        let fill = OrderEvent::Fill {
            order_id: "a".to_string(),
            execution_id: "e1".to_string(),
            side: Action::Buy,
            price: dec!(100),
            quantity: dec!(5),
            cumulative_quantity: dec!(5),
            fee: Decimal::ZERO,
            fee_asset: crate::types::fee::FeeAsset::Quote,
            timestamp: 1,
        };
        policy.on_order_event(&fill).unwrap();
        policy.on_order_event(&fill).unwrap();
        assert_eq!(policy.position(), dec!(5));
        let sell = PretradeRequest::new(
            "s",
            Action::Sell,
            Some(dec!(100)),
            dec!(20),
            OrderPurpose::Quote,
            2,
        );
        assert_eq!(policy.check(&sell).approved_quantity(dec!(20)), dec!(15));
    }

    #[test]
    fn test_reset_restores_the_initial_breaker() {
        let config =
            CircuitBreakerConfig::new(dec!(1000), dec!(0.05), 5, dec!(0.10), 300_000, 60_000)
                .unwrap();
        let mut policy = policy(dec!(10)).with_circuit_breaker(CircuitBreaker::new(config));
        policy.circuit_breaker_mut().unwrap().trigger_manual(0);
        assert!(policy.check(&buy("a", dec!(1))).is_rejected());
        policy.reset();
        assert_eq!(policy.check(&buy("b", dec!(1))), PretradeDecision::Accepted);
    }

    #[test]
    fn test_bound_instrument_rejects_others() {
        let mut policy = policy(dec!(10)).with_instrument("BTC-USD");
        let eth = buy("e", dec!(1)).with_instrument("ETH-USD");
        assert!(policy.check(&eth).is_rejected());
        let btc = buy("b", dec!(1)).with_instrument("BTC-USD");
        assert_eq!(policy.check(&btc), PretradeDecision::Accepted);
        assert_eq!(policy.check(&buy("n", dec!(1))), PretradeDecision::Accepted);
    }

    #[test]
    fn test_duplicate_and_invalid_requests_are_rejected() {
        let mut policy = policy(dec!(10));
        assert_eq!(policy.check(&buy("a", dec!(1))), PretradeDecision::Accepted);
        assert!(policy.check(&buy("a", dec!(1))).is_rejected());
        assert!(policy.check(&buy("z", Decimal::ZERO)).is_rejected());
    }

    fn other(id: &str, quantity: Decimal, timestamp: u64) -> PretradeRequest {
        PretradeRequest::new(
            id,
            Action::Other,
            Some(dec!(100)),
            quantity,
            OrderPurpose::Quote,
            timestamp,
        )
    }

    #[test]
    fn test_check_unsupported_action_is_rejected_without_state_change() {
        let mut policy = policy(dec!(10));
        assert_eq!(
            policy.check(&other("a", dec!(5), 0)),
            PretradeDecision::Rejected {
                reason: "unsupported action Other: only buy and sell are supported".to_string(),
            }
        );
        assert_eq!(policy.live_quantity(Action::Buy).unwrap(), Decimal::ZERO);
        assert_eq!(policy.live_quantity(Action::Sell).unwrap(), Decimal::ZERO);
        // The id was not reserved.
        assert_eq!(policy.check(&buy("a", dec!(5))), PretradeDecision::Accepted);
    }

    #[test]
    fn test_check_unsupported_action_leaves_the_breaker_untouched() {
        let config =
            CircuitBreakerConfig::new(dec!(1000), dec!(0.05), 5, dec!(0.10), 300_000, 60_000)
                .unwrap();
        let mut policy = policy(dec!(10)).with_circuit_breaker(CircuitBreaker::new(config));
        let breaker = policy.circuit_breaker_mut().unwrap();
        breaker.trigger_manual(0);
        breaker.start_cooldown(0);
        let later = 10_000_000;

        assert!(policy.check(&other("a", dec!(1), later)).is_rejected());
        assert!(policy.circuit_breaker_mut().unwrap().state().is_cooldown());

        // A supported order at the same time runs the cooldown check.
        let mut request = buy("b", dec!(1));
        request.timestamp = later;
        assert_eq!(policy.check(&request), PretradeDecision::Accepted);
        assert!(!policy.circuit_breaker_mut().unwrap().state().is_cooldown());
    }

    #[test]
    fn test_live_quantity_unsupported_action_returns_err() {
        let policy = policy(dec!(10));
        assert_eq!(
            policy.live_quantity(Action::Other),
            Err(crate::types::error::MMError::UnsupportedAction {
                action: Action::Other
            })
        );
    }

    fn other_fill(order_id: &str, quantity: Decimal) -> OrderEvent {
        OrderEvent::Fill {
            order_id: order_id.to_string(),
            execution_id: "e1".to_string(),
            side: Action::Other,
            price: dec!(100),
            quantity,
            cumulative_quantity: quantity,
            fee: Decimal::ZERO,
            fee_asset: crate::types::fee::FeeAsset::Quote,
            timestamp: 1,
        }
    }

    #[test]
    fn test_fill_with_unsupported_action_latches_a_fault() {
        let mut policy = policy(dec!(10));
        assert_eq!(policy.check(&buy("a", dec!(5))), PretradeDecision::Accepted);
        let unsupported = crate::types::error::MMError::UnsupportedAction {
            action: Action::Other,
        };

        assert_eq!(
            policy.on_order_event(&other_fill("a", dec!(2))),
            Err(unsupported.clone())
        );
        assert_eq!(policy.fault(), Some(&unsupported));
        // Nothing was booked.
        assert_eq!(policy.position(), Decimal::ZERO);
        assert_eq!(policy.live_quantity(Action::Buy).unwrap(), dec!(5));
        // Every order is rejected while the fault is latched.
        assert_eq!(
            policy.check(&buy("b", dec!(1))),
            PretradeDecision::Rejected {
                reason: "policy fault: unsupported action Other: only buy and sell are supported"
                    .to_string(),
            }
        );

        // The execution id was not consumed: a corrected redelivery applies,
        // but the fault stays until reset.
        let mut corrected = other_fill("a", dec!(2));
        if let OrderEvent::Fill { side, .. } = &mut corrected {
            *side = Action::Buy;
        }
        policy.on_order_event(&corrected).unwrap();
        assert_eq!(policy.position(), dec!(2));
        assert!(policy.check(&buy("c", dec!(1))).is_rejected());

        policy.reset();
        assert_eq!(policy.fault(), None);
        assert_eq!(policy.check(&buy("d", dec!(1))), PretradeDecision::Accepted);
    }

    /// Regression: a real fill of unknown direction must not leave
    /// the policy approving orders against an understated position.
    #[test]
    fn test_unknown_direction_fill_does_not_fail_open() {
        let mut policy = policy(dec!(10));
        assert_eq!(
            policy.check(&buy("a", dec!(10))),
            PretradeDecision::Accepted
        );
        assert!(policy.on_order_event(&other_fill("a", dec!(10))).is_err());
        policy
            .on_order_event(&OrderEvent::Canceled {
                order_id: "a".to_string(),
            })
            .unwrap();
        assert!(policy.check(&buy("b", dec!(10))).is_rejected());
    }
}
