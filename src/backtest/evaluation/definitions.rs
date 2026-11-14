//! What each reported number means.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Unit of a metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum MetricUnit {
    /// Quote asset.
    Quote,
    /// Quote asset per unit of base asset.
    QuotePerBase,
    /// Base asset.
    Base,
    /// A count.
    Count,
    /// A dimensionless ratio.
    Ratio,
}

/// How to read the sign of a metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum Direction {
    /// Positive is good for the strategy.
    PositiveIsGain,
    /// Positive is bad for the strategy.
    PositiveIsCost,
    /// Never negative; larger is neither good nor bad by itself.
    Magnitude,
    /// Never negative; larger is worse.
    MagnitudeIsCost,
}

/// Over which time a metric is measured.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum MetricHorizon {
    /// The whole window.
    Window,
    /// At the fill, against the mark in force.
    AtFill,
    /// A declared time after the fill; [`MetricValue`](super::MetricValue)
    /// carries it.
    AfterFill,
}

/// The capital a metric is divided by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum CapitalDenominator {
    /// Not divided by capital.
    None,
    /// The capital declared for the window.
    DeclaredCapital,
}

/// How values are combined over the window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum Aggregation {
    /// A sum.
    Sum,
    /// A mean weighted by fill size.
    VolumeWeightedMean,
    /// A mean weighted by time between the first and the last mark, not per tick.
    TimeWeightedMean,
    /// The largest value.
    Maximum,
    /// A count.
    Count,
    /// A quotient of two other metrics.
    Quotient,
}

/// The declared meaning of a metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MetricDefinition {
    /// Unit.
    pub unit: MetricUnit,
    /// Sign.
    pub direction: Direction,
    /// Horizon.
    pub horizon: MetricHorizon,
    /// Capital denominator.
    pub capital_denominator: CapitalDenominator,
    /// Aggregation over the window.
    pub aggregation: Aggregation,
}

/// The metrics of a window.
///
/// `s` is `+1` for one of our buys and `-1` for one of our sells, so `-s` is the
/// taker's direction on a passive fill. `P` is the fill price, `M_f` the mark in
/// force at the execution time of the fill and `M_H` the first mark, the last of any tied at its time, at or after
/// the execution time plus the horizon.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum MetricId {
    /// `spread capture + inventory - quote fees - base fees`.
    NetPnl,
    /// `Σ s q (M_f - P)`.
    SpreadCapture,
    /// `Σ p_i (M_i - M_{i-1})`.
    InventoryPnl,
    /// Fees fixed in the quote asset; negative for rebates.
    QuoteFees,
    /// Fees in the base asset, valued at `M_f`.
    BaseFees,
    /// Costs of a hedge in a second instrument, [`HedgeAttribution::costs`]: not
    /// modelled with one instrument.
    ///
    /// [`HedgeAttribution::costs`]: super::HedgeAttribution::costs
    HedgeCosts,
    /// `s (M_H - P)` per unit, our result on the fill after the horizon.
    SignedMarkout,
    /// `2 (-s) (P - M_H)` per unit, the published form with the taker's direction.
    /// On our passive fills it is twice [`MetricId::SignedMarkout`], with the
    /// same sign.
    RealizedSpread,
    /// `2 (-s) (P - M_f)` per unit. It equals the realized spread plus the price
    /// impact.
    EffectiveSpread,
    /// `2 (-s) (M_H - M_f)` per unit: positive when the mid moved the way the taker
    /// traded, which is against us.
    PriceImpact,
    /// Number of fills.
    Fills,
    /// Total size of the fills, gross of base fees.
    Volume,
    /// Time weighted mean of the absolute position.
    MeanAbsInventory,
    /// Largest absolute position, net of base fees.
    MaxAbsInventory,
    /// Largest fall of `cash + position * mark - quote fees` from its running
    /// peak, sampled at each mark, the first peak being the flat start.
    MaxDrawdown,
    /// [`MetricId::NetPnl`] over the capital.
    NetReturnOnCapital,
    /// [`MetricId::MaxDrawdown`] over the capital.
    MaxDrawdownOnCapital,
    /// Funding received on any lane; negative when paid.
    Funding,
    /// `Σ (X + s c M M_f)` over position changes that are not fills, such as an
    /// auto deleveraging close, `X` the signed gross cash of the close as given.
    TransferEdge,
    /// Account cash outside every lane, such as an auto deleveraging top up or an
    /// insurance fund draw; received is positive.
    ExternalFlows,
    /// Settlement cash minus `p M S`, the exact value of the settled position.
    SettlementRounding,
    /// Orders, liquidation orders left out.
    Orders,
    /// Orders with at least one fill, liquidation orders left out.
    FilledOrders,
    /// Size offered over the orders, in units of the underlying (contracts times
    /// the multiplier): the size first accepted plus every positive increase of
    /// the open quantity by an accepted replace.
    AcceptedQuantity,
    /// Size filled over the orders, in units of the underlying.
    FilledQuantity,
    /// [`MetricId::FilledOrders`] over [`MetricId::Orders`].
    FillRatioByCount,
    /// [`MetricId::FilledQuantity`] over [`MetricId::AcceptedQuantity`].
    FillRatioByQuantity,
    /// Largest absolute difference of an attributed net result, of a lane or of
    /// the account, to the books.
    ReconciliationResidual,
}

impl MetricId {
    /// Every metric.
    pub const ALL: [Self; 28] = [
        Self::NetPnl,
        Self::SpreadCapture,
        Self::InventoryPnl,
        Self::QuoteFees,
        Self::BaseFees,
        Self::HedgeCosts,
        Self::SignedMarkout,
        Self::RealizedSpread,
        Self::EffectiveSpread,
        Self::PriceImpact,
        Self::Fills,
        Self::Volume,
        Self::MeanAbsInventory,
        Self::MaxAbsInventory,
        Self::MaxDrawdown,
        Self::NetReturnOnCapital,
        Self::MaxDrawdownOnCapital,
        Self::Funding,
        Self::TransferEdge,
        Self::ExternalFlows,
        Self::SettlementRounding,
        Self::Orders,
        Self::FilledOrders,
        Self::AcceptedQuantity,
        Self::FilledQuantity,
        Self::FillRatioByCount,
        Self::FillRatioByQuantity,
        Self::ReconciliationResidual,
    ];

    /// The metrics [`WindowMetrics::values`](super::WindowMetrics::values) reports, in order.
    pub const WINDOW: [Self; 17] = [
        Self::NetPnl,
        Self::SpreadCapture,
        Self::InventoryPnl,
        Self::QuoteFees,
        Self::BaseFees,
        Self::HedgeCosts,
        Self::SignedMarkout,
        Self::RealizedSpread,
        Self::EffectiveSpread,
        Self::PriceImpact,
        Self::Fills,
        Self::Volume,
        Self::MeanAbsInventory,
        Self::MaxAbsInventory,
        Self::MaxDrawdown,
        Self::NetReturnOnCapital,
        Self::MaxDrawdownOnCapital,
    ];

    /// The metrics only an account of many lanes reports, in order: [`AccountAttribution::values`](super::AccountAttribution::values) and
    /// [`LaneOutcome::values`](super::LaneOutcome::values).
    pub const ACCOUNT: [Self; 11] = [
        Self::Funding,
        Self::TransferEdge,
        Self::ExternalFlows,
        Self::SettlementRounding,
        Self::Orders,
        Self::FilledOrders,
        Self::AcceptedQuantity,
        Self::FilledQuantity,
        Self::FillRatioByCount,
        Self::FillRatioByQuantity,
        Self::ReconciliationResidual,
    ];

    /// The declared meaning of this metric.
    #[must_use]
    pub const fn definition(self) -> MetricDefinition {
        use Aggregation as A;
        use CapitalDenominator as C;
        use Direction as D;
        use MetricHorizon as H;
        use MetricUnit as U;
        let (unit, direction, horizon, capital_denominator, aggregation) = match self {
            Self::NetPnl | Self::InventoryPnl => {
                (U::Quote, D::PositiveIsGain, H::Window, C::None, A::Sum)
            }
            Self::SpreadCapture => (U::Quote, D::PositiveIsGain, H::AtFill, C::None, A::Sum),
            Self::QuoteFees | Self::BaseFees | Self::HedgeCosts => {
                (U::Quote, D::PositiveIsCost, H::Window, C::None, A::Sum)
            }
            Self::SignedMarkout | Self::RealizedSpread => (
                U::QuotePerBase,
                D::PositiveIsGain,
                H::AfterFill,
                C::None,
                A::VolumeWeightedMean,
            ),
            Self::EffectiveSpread => (
                U::QuotePerBase,
                D::PositiveIsGain,
                H::AtFill,
                C::None,
                A::VolumeWeightedMean,
            ),
            Self::PriceImpact => (
                U::QuotePerBase,
                D::PositiveIsCost,
                H::AfterFill,
                C::None,
                A::VolumeWeightedMean,
            ),
            Self::Fills => (U::Count, D::Magnitude, H::Window, C::None, A::Count),
            Self::Volume => (U::Base, D::Magnitude, H::Window, C::None, A::Sum),
            Self::MeanAbsInventory => (
                U::Base,
                D::Magnitude,
                H::Window,
                C::None,
                A::TimeWeightedMean,
            ),
            Self::MaxAbsInventory => (U::Base, D::Magnitude, H::Window, C::None, A::Maximum),
            Self::MaxDrawdown => (U::Quote, D::MagnitudeIsCost, H::Window, C::None, A::Maximum),
            Self::NetReturnOnCapital => (
                U::Ratio,
                D::PositiveIsGain,
                H::Window,
                C::DeclaredCapital,
                A::Quotient,
            ),
            Self::MaxDrawdownOnCapital => (
                U::Ratio,
                D::MagnitudeIsCost,
                H::Window,
                C::DeclaredCapital,
                A::Quotient,
            ),
            Self::Funding | Self::ExternalFlows | Self::SettlementRounding => {
                (U::Quote, D::PositiveIsGain, H::Window, C::None, A::Sum)
            }
            Self::TransferEdge => (U::Quote, D::PositiveIsGain, H::AtFill, C::None, A::Sum),
            Self::Orders | Self::FilledOrders => {
                (U::Count, D::Magnitude, H::Window, C::None, A::Count)
            }
            Self::AcceptedQuantity | Self::FilledQuantity => {
                (U::Base, D::Magnitude, H::Window, C::None, A::Sum)
            }
            Self::FillRatioByCount | Self::FillRatioByQuantity => {
                (U::Ratio, D::Magnitude, H::Window, C::None, A::Quotient)
            }
            Self::ReconciliationResidual => {
                (U::Quote, D::MagnitudeIsCost, H::Window, C::None, A::Maximum)
            }
        };
        MetricDefinition {
            unit,
            direction,
            horizon,
            capital_denominator,
            aggregation,
        }
    }
}
