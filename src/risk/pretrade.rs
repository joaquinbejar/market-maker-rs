//! Pretrade risk port.
//!
//! Every order passes a [`PretradePolicy`] before it is submitted. The
//! policy accepts it, reduces its quantity or rejects it, and is told about
//! the order's lifecycle afterwards (accepts, fills, cancels) so it can track
//! live exposure and reservations. The same seam serves the quote path of
//! the backtest engine
//! ([`BacktestEngine::with_pretrade_policy`](crate::backtest::BacktestEngine::with_pretrade_policy))
//! and the live hedge or quote path of
//! [`OrderManager::register_order_checked`](crate::execution::OrderManager::register_order_checked).
//!
//! [`LimitsPolicy`] is a local policy built on [`RiskLimits`] and
//! [`CircuitBreaker`]. With the `margin` feature, the backtest module offers
//! an adapter over the `margin-rs` risk gate.
//!
//! # Example
//!
//! ```rust
//! use market_maker_rs::dec;
//! use market_maker_rs::execution::Side;
//! use market_maker_rs::risk::RiskLimits;
//! use market_maker_rs::risk::pretrade::{
//!     LimitsPolicy, OrderPurpose, PretradeDecision, PretradePolicy, PretradeRequest,
//! };
//!
//! let limits = RiskLimits::new(dec!(10), dec!(1_000_000), dec!(0)).unwrap();
//! let mut policy = LimitsPolicy::new(limits);
//! let request = PretradeRequest::new("o-1", Side::Buy, Some(dec!(100)), dec!(15), OrderPurpose::Quote, 0);
//! let decision = policy.check(&request);
//! assert_eq!(decision.approved_quantity(request.quantity), dec!(10));
//! assert!(matches!(decision, PretradeDecision::Reduced { .. }));
//! ```

use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::Decimal;
use crate::execution::Side;

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
    /// Order side.
    pub side: Side,
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
        side: Side,
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
        /// Order side.
        side: Side,
        /// Execution price.
        price: Decimal,
        /// Quantity of this execution.
        quantity: Decimal,
        /// Cumulative filled quantity of the order after this execution.
        cumulative_quantity: Decimal,
        /// Fee in the quote asset.
        fee: Decimal,
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
///   [`PretradePolicy::on_order_event`], in venue order.
/// - [`PretradePolicy::reset`] returns the policy to its initial state.
pub trait PretradePolicy: fmt::Debug {
    /// Decides on an order before submission.
    fn check(&mut self, request: &PretradeRequest) -> PretradeDecision;

    /// Market mark observed at `timestamp`.
    fn on_market_data(&mut self, _mark: Decimal, _timestamp: u64) {}

    /// Lifecycle event of an order.
    fn on_order_event(&mut self, _event: &OrderEvent) {}

    /// Returns to the initial state.
    fn reset(&mut self);

    /// Human readable description.
    fn description(&self) -> String {
        "custom pretrade policy".to_string()
    }
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
#[derive(Debug, Clone)]
pub struct LimitsPolicy {
    limits: RiskLimits,
    instrument: Option<String>,
    initial_breaker: Option<CircuitBreaker>,
    breaker: Option<CircuitBreaker>,
    executions: HashSet<String>,
    position: Decimal,
    mark: Option<Decimal>,
    live: HashMap<String, (Side, Decimal)>,
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

    /// Position implied by the fills seen so far.
    #[must_use]
    pub fn position(&self) -> Decimal {
        self.position
    }

    /// Remaining quantity of live orders on `side`.
    #[must_use]
    pub fn live_quantity(&self, side: Side) -> Decimal {
        self.live
            .values()
            .filter(|(s, _)| *s == side)
            .map(|(_, q)| *q)
            .sum()
    }

    fn signed(side: Side, quantity: Decimal) -> Decimal {
        match side {
            Side::Buy => quantity,
            Side::Sell => -quantity,
        }
    }
}

impl PretradePolicy for LimitsPolicy {
    fn check(&mut self, request: &PretradeRequest) -> PretradeDecision {
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

        // Worst case exposure on the order's side before this order.
        let worst = self.position + Self::signed(request.side, self.live_quantity(request.side));
        let mut capacity = request.quantity;
        let by_position = match request.side {
            Side::Buy => self.limits.max_position - worst,
            Side::Sell => self.limits.max_position + worst,
        };
        capacity = capacity.min(by_position);
        // An unrepresentable quotient (tiny price) means the notional limit
        // does not bind.
        if let Some(units) = self.limits.max_notional.checked_div(price) {
            let by_notional = match request.side {
                Side::Buy => units - worst,
                Side::Sell => units + worst,
            };
            capacity = capacity.min(by_notional);
        }

        let decision = if capacity >= request.quantity {
            PretradeDecision::Accepted
        } else if capacity > Decimal::ZERO {
            PretradeDecision::Reduced {
                quantity: capacity,
                reason: "position or notional limit".to_string(),
            }
        } else {
            PretradeDecision::Rejected {
                reason: "position or notional limit reached".to_string(),
            }
        };
        let approved = decision.approved_quantity(request.quantity);
        if approved > Decimal::ZERO {
            self.live
                .insert(request.order_id.clone(), (request.side, approved));
        }
        decision
    }

    fn on_market_data(&mut self, mark: Decimal, _timestamp: u64) {
        if mark > Decimal::ZERO {
            self.mark = Some(mark);
        }
    }

    fn on_order_event(&mut self, event: &OrderEvent) {
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
                ..
            } => {
                // A replayed execution changes nothing.
                if !self.executions.insert(execution_id.clone()) {
                    return;
                }
                self.position += Self::signed(*side, *quantity);
                if let Some((_, remaining)) = self.live.get_mut(order_id) {
                    *remaining -= *quantity;
                    if *remaining <= Decimal::ZERO {
                        self.live.remove(order_id);
                    }
                }
            }
        }
    }

    fn reset(&mut self) {
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
            Side::Buy,
            Some(dec!(100)),
            quantity,
            OrderPurpose::Quote,
            0,
        )
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
        policy.on_order_event(&OrderEvent::Canceled {
            order_id: "b".to_string(),
        });
        policy.on_order_event(&OrderEvent::Fill {
            order_id: "a".to_string(),
            execution_id: "e1".to_string(),
            side: Side::Buy,
            price: dec!(100),
            quantity: dec!(6),
            cumulative_quantity: dec!(6),
            fee: Decimal::ZERO,
            timestamp: 1,
        });
        assert_eq!(policy.position(), dec!(6));
        assert_eq!(policy.live_quantity(Side::Buy), Decimal::ZERO);
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
            Side::Sell,
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
        let market = PretradeRequest::new("m", Side::Buy, None, dec!(1), OrderPurpose::Hedge, 0);
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
            Side::Buy,
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
            side: Side::Buy,
            price: dec!(100),
            quantity: dec!(5),
            cumulative_quantity: dec!(5),
            fee: Decimal::ZERO,
            timestamp: 1,
        };
        policy.on_order_event(&fill);
        policy.on_order_event(&fill);
        assert_eq!(policy.position(), dec!(5));
        let sell = PretradeRequest::new(
            "s",
            Side::Sell,
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
}
