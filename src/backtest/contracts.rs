//! A backtest of several dated futures and perpetuals in one run, against one `margin-rs`
//! ledger.
//!
//! [`ContractEngine`] is the path for a run across contracts, such as a roll, which
//! [`BacktestEngine::with_instrument`](super::BacktestEngine::with_instrument) does not
//! take: its ticks, quotes and orders name no contract. Each contract has its own data,
//! its own fill model and its own orders, and all of them share one clock, one ledger
//! and one policy. Orders are one sided and in contracts, and a strategy may answer any
//! report with new commands.
//!
//! # What is simulated
//!
//! Matching, latency, expiry, a declared cash settlement and declared perpetual funding.
//! There is no margin
//! requirement and no liquidation: fills book with overdraft, reported in
//! [`ContractRunResult::overdrafts`], and a pretrade policy in contracts is no simulation
//! of solvency.
//!
//! # Funding
//!
//! [`ContractEngine::with_perpetual`] takes a perpetual with its declared funding. At each
//! instant in the run the position pays or receives its notional at the declared mark times
//! the declared rate, rounded half to even once at the settlement asset's scale, and books
//! through the ledger, with overdraft if the balance does not cover it. An instant runs
//! before anything at that time can match, so a position pays the instants in
//! `(opened, closed]`. Only instants after the first tick count, as what came before belongs
//! to no position this run knows the age of.
//!
//! Funding is in [`ContractRunResult::total_pnl`] and apart in
//! [`ContractRunResult::total_funding`], which
//! [`ContractRunResult::realized_pnl`] excludes. An instant of the run with no declared rate
//! is in [`ContractSummary::funding_missing`] and makes the run not consistent. A strategy is
//! not told of funding, and funding declared past the end of the clock does not book. The
//! clock runs on past the data to the latest declared settlement, so funding up to then
//! books, on a position nothing can then change, and its instants count as missing if no
//! rate was declared, whether a position was open or not.
//!
//! # Time
//!
//! Ticks of all contracts are merged by time, then by the order the contracts were
//! registered in. The clock never goes back: an out of order tick is processed at the
//! current time, and matches by its own time, as in the single contract engine. An order
//! may only match a tick strictly after it became active, so one sent on a tick of
//! another contract at the same time waits for a later tick.
//!
//! At one time, events run in this order: funding, expiries, venue messages, settlements, report
//! deliveries, a strategy's wake, observations. So nothing executes on a contract at its
//! expiry. A strategy that must act at a time with no tick says so in
//! [`ContractStrategy::next_wake_ms`], and every callback has the time of the decision in
//! [`ContractView::now_ms`], which a report's or a tick's own time may be well behind.
//!
//! # Execution and delivery
//!
//! A fill books in the ledger when it matches, at the tick's time, and only then enters
//! the report channel, which is first in, first out. So a fill matched before an expiry
//! is in the ledger before that contract's settlement, however late it is delivered, and
//! the strategy hears of it before it hears of the settlement. This engine therefore never
//! produces a fill closed at the settlement price by the ledger.
//!
//! The policy sees a fill before the ledger does, as in the single contract engine. A
//! command returned by any callback runs after that callback returns and through the same
//! checks as any other. So a strategy could answer each rejection with a new order at zero
//! latency for ever: past [`MAX_COMMANDS_PER_INSTANT`] commands at one time, that is a fault
//! and they are discarded, so the run ends.
//!
//! # Expiry and settlement
//!
//! When the clock reaches a contract's expiry, the venue cancels its live orders and those
//! on their way, new ones are rejected locally, and its ticks are counted and ignored, so
//! they do not move its mark. Its position stays at the last mark until a declared
//! [`SettlementPrice`] is published, when it books through
//! [`FutureTerms::settle_in_ledger`] and the strategy hears of it through the report
//! channel. Nothing is inferred: with no declared price it stays open at that mark,
//! [`ContractSummary::expired_unsettled`] says so, the run is not consistent, and
//! [`ContractRunResult::stale_from_ms`] gives the first equity that rests on such a mark.
//! A contract whose data ended before its expiry is simply open.
//!
//! # When the data ends
//!
//! The clock then runs on to the latest declared settlement, so a settlement is not lost
//! for want of a tick. No liquidity is invented: nothing matches. Observations still on
//! their way are dropped, reports still on their way are delivered, and any command
//! returned from then on is discarded and counted in
//! [`ContractRunResult::discarded_commands`].
//!
//! # Valuation
//!
//! Each contract's mark is the mid of its latest tick, so equity mixes marks of different
//! ages when a contract ticks rarely; [`ContractSummary::last_mark_ms`] gives the age.
//!
//! Equity is `initial_capital + realized + unrealized - fees`, in the settlement asset,
//! where realized and fees count from the start of the run: what the ledger already held
//! is not counted again. A position the ledger started with has its whole unrealized PnL
//! counted, from its own cost.
//!
//! Until each open contract has a mark there is no valuation, no equity point and no fault,
//! only [`ContractRunResult::unvalued_ticks`]. A contract still open with no mark when the
//! run ends makes it not consistent. A valuation that fails on a mark it has, or a figure
//! that overflows, is a fault counted in `valuation_failures`: the previous figures stay,
//! new orders stop, and no limit value stands in for a result. The single contract engine
//! saturates its sums instead.
//!
//! # Against the single contract engine
//!
//! With one contract and a strategy on the grid, trades, reports, fees, PnL and the equity
//! curve are the same, and a test holds this. They differ in these ways:
//!
//! - it refuses a price or a size off the contract's grid, where that engine rounds;
//! - it does not match at or after an expiry, where that engine's accountant refuses the
//!   fill and records it;
//! - realized and fees count from the start of the run, where that engine counts what the
//!   ledger already held;
//! - a figure that overflows is a fault, where that engine saturates it;
//! - it knows the position the ledger started with, where a strategy there starts from zero.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::panic::{RefUnwindSafe, UnwindSafe};

use financial_types::instrument::InstrumentId;
use margin_rs::ledger::{AssetPnl, Booking, Execution, FundingPayment, Ledger};
use margin_rs::{Asset, ExecutionId};

use super::accounting::largest_execution_number;
use super::data::{HistoricalDataSource, MarketTick};
use super::engine::{
    AccountingStatus, FeeCharge, MAX_RECORDED_FAULTS, PretradeRecord, SlippageModel,
    fill_increment, next_due, settlement_charge, sharpe_ratio, slipped_price,
};
use super::fill_models::{FillModel, SimulatedOrder};
use super::latency::{JitterRng, SimLatencyConfig};
use super::lifecycle::{ExecutionKind, ExecutionReport, SimClock, SimOrder, SimOrderId};
use crate::Decimal;
use crate::execution::Action;
use crate::instruments::futures::{FutureTerms, SettlementPrice, identify_spec_in_ledger};
use crate::instruments::perpetual::PerpetualTerms;
use crate::risk::pretrade::{OrderEvent, OrderPurpose, PretradePolicy, PretradeRequest};
use crate::types::action::Dir;
use crate::types::decimal::OrOverflow;
use crate::types::error::{MMError, MMResult};
use crate::types::fee::FeeAsset;

/// The commands a run carries out at one time, past which they are a fault and are
/// discarded: see the [module documentation](self).
pub const MAX_COMMANDS_PER_INSTANT: u64 = 100_000;

/// The faults taken from a strategy after one callback.
pub const MAX_STRATEGY_FAULTS_PER_CALLBACK: usize = 64;

/// Configuration of a [`ContractEngine`].
#[derive(Debug, Clone)]
pub struct ContractConfig {
    /// Initial capital, in the settlement asset.
    pub initial_capital: Decimal,
    /// The fee rate on a contract's notional: negative is a rebate.
    pub fee_rate: Decimal,
    /// Decimal places of the settlement asset for a fee, rounded half to even once.
    pub fee_scale: u32,
    /// Slippage model.
    pub slippage: SlippageModel,
    /// Record equity at each tick.
    pub record_equity_curve: bool,
    /// Record all trades and reports.
    pub record_trades: bool,
}

impl ContractConfig {
    /// A configuration with no slippage that records everything.
    #[must_use]
    pub fn new(initial_capital: Decimal, fee_rate: Decimal, fee_scale: u32) -> Self {
        Self {
            initial_capital,
            fee_rate,
            fee_scale,
            slippage: SlippageModel::None,
            record_equity_curve: true,
            record_trades: true,
        }
    }
}

/// A one sided limit order for one contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContractOrder {
    /// The strategy's own identifier, unique in a run, which its reports carry.
    pub tag: u64,
    /// The contract.
    pub contract: InstrumentId,
    /// Buy or sell.
    pub side: Action,
    /// The limit price, on the contract's grid.
    pub price: Decimal,
    /// The contracts, on the contract's lot.
    pub quantity: Decimal,
    /// Why it is sent.
    pub purpose: OrderPurpose,
}

/// What a strategy asks of the venue.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContractCommand {
    /// Send an order. A repeated tag or a contract not registered is a fault.
    Submit(ContractOrder),
    /// Cancel an order. If it already ended, as when a cancel crosses a fill, a
    /// [`ExecutionKind::CancelRejected`] answers it. A tag never seen is a fault.
    Cancel {
        /// The order's tag.
        tag: u64,
    },
    /// Cancel every live or pending order of a contract.
    CancelAll {
        /// The contract.
        contract: InstrumentId,
    },
}

/// A transition of an order, with what the engine knows of it.
#[derive(Debug, Clone, PartialEq)]
pub struct ContractReport {
    /// The contract.
    pub contract: InstrumentId,
    /// The order's tag.
    pub tag: u64,
    /// The transition.
    pub report: ExecutionReport,
}

/// One execution.
#[derive(Debug, Clone, PartialEq)]
pub struct ContractFill {
    /// The contract.
    pub contract: InstrumentId,
    /// The order's tag.
    pub tag: u64,
    /// The identifier in the ledger's journal.
    pub execution_id: String,
    /// Buy or sell.
    pub side: Action,
    /// The price, slippage included.
    pub price: Decimal,
    /// The contracts.
    pub quantity: Decimal,
    /// The order's cumulative contracts, this execution included.
    pub cumulative_quantity: Decimal,
    /// The fee in the settlement asset: negative is a rebate.
    pub fee: Decimal,
    /// When it executed, in milliseconds.
    pub timestamp: u64,
}

/// What a strategy knows of the positions: those the ledger started with, plus fills
/// delivered to it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ContractView {
    positions: Vec<(InstrumentId, Decimal)>,
    now_ms: u64,
}

impl ContractView {
    /// The time of this decision, in milliseconds: when the event reached the strategy.
    #[must_use]
    pub fn now_ms(&self) -> u64 {
        self.now_ms
    }

    /// The position on `contract`, in contracts.
    #[must_use]
    pub fn position(&self, contract: &InstrumentId) -> Decimal {
        self.positions
            .iter()
            .find(|(id, _)| id == contract)
            .map_or(Decimal::ZERO, |(_, position)| *position)
    }
}

/// A strategy over several contracts. Every callback may return commands.
pub trait ContractStrategy {
    /// A tick of `contract` is seen, after its resting orders were matched against it.
    fn on_tick(
        &mut self,
        contract: &InstrumentId,
        tick: &MarketTick,
        view: &ContractView,
    ) -> Vec<ContractCommand>;

    /// An order changed.
    fn on_report(
        &mut self,
        _report: &ContractReport,
        _view: &ContractView,
    ) -> Vec<ContractCommand> {
        Vec::new()
    }

    /// An order executed, after its report.
    fn on_fill(&mut self, _fill: &ContractFill, _view: &ContractView) -> Vec<ContractCommand> {
        Vec::new()
    }

    /// `contract` settled at `price`, closing `contracts`, signed like the position.
    fn on_settlement(
        &mut self,
        _contract: &InstrumentId,
        _price: &SettlementPrice,
        _contracts: Decimal,
        _view: &ContractView,
    ) -> Vec<ContractCommand> {
        Vec::new()
    }

    /// When to call [`ContractStrategy::on_wake`], which the engine asks after each
    /// callback and at the start of a run. A time before the clock, or one it already woke
    /// the strategy at, is ignored. At the time itself it runs after the deliveries of
    /// that time, so a report arriving then does not take its place.
    fn next_wake_ms(&self) -> Option<u64> {
        None
    }

    /// The time asked for in [`ContractStrategy::next_wake_ms`] came.
    fn on_wake(&mut self, _view: &ContractView) -> Vec<ContractCommand> {
        Vec::new()
    }

    /// What went wrong in the strategy, which the engine asks after each callback and
    /// before its commands, as a fault: new orders then stop, and the run is not consistent.
    /// At most [`MAX_STRATEGY_FAULTS_PER_CALLBACK`] are taken each time.
    fn take_fault(&mut self) -> Option<MMError> {
        None
    }

    /// Returns to the initial state.
    fn reset(&mut self);
}

/// How one contract ended a run.
#[derive(Debug, Clone, PartialEq)]
pub struct ContractSummary {
    /// The contract.
    pub contract: InstrumentId,
    /// The position in the ledger, in contracts.
    pub position: Decimal,
    /// The largest absolute position in this run, from the one the ledger started with.
    pub max_position: Decimal,
    /// The executions of this run.
    pub fills: u64,
    /// The latest mark.
    pub last_mark: Option<Decimal>,
    /// The time of that mark.
    pub last_mark_ms: Option<u64>,
    /// The clock reached its expiry.
    pub expired: bool,
    /// A declared settlement booked.
    pub settled: bool,
    /// A declared settlement that the ledger refused when it was published.
    pub settlement_failed: bool,
    /// Past its expiry, with a position and no settlement: it rests on its last mark.
    pub expired_unsettled: bool,
    /// Its ticks ignored, as they came after its expiry.
    pub ignored_ticks: u64,
    /// Funding on this contract in this run: positive is received.
    pub funding: Decimal,
    /// Funding instants of the run with no declared rate.
    pub funding_missing: u64,
}

/// One funding instant a position was open on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContractFunding {
    /// The contract.
    pub contract: InstrumentId,
    /// The instant, in milliseconds.
    pub at_ms: u64,
    /// The position that paid or received, in contracts.
    pub position: Decimal,
    /// The declared mark.
    pub mark: Decimal,
    /// The declared rate.
    pub rate: Decimal,
    /// The amount: positive is received. Zero if it rounded to nothing, with no ledger entry.
    pub amount: Decimal,
}

/// The result of a [`ContractEngine`] run.
#[derive(Debug, Clone, Default)]
pub struct ContractRunResult {
    /// PnL before fees in this run, funding included.
    pub total_pnl: Decimal,
    /// The realized part of [`ContractRunResult::total_pnl`], without funding.
    pub realized_pnl: Decimal,
    /// The funding part of [`ContractRunResult::total_pnl`]: positive is received.
    pub total_funding: Decimal,
    /// Each funding instant a position was open on, if trades are recorded.
    pub fundings: Vec<ContractFunding>,
    /// Fees paid in this run.
    pub total_fees: Decimal,
    /// `total_pnl - total_fees`.
    pub net_pnl: Decimal,
    /// The unrealized part of [`ContractRunResult::total_pnl`].
    pub unrealized_pnl: Decimal,
    /// Executions.
    pub num_trades: u64,
    /// Ticks read, on all contracts.
    pub num_ticks: u64,
    /// The first tick's time.
    pub start_time: u64,
    /// The last tick's time.
    pub end_time: u64,
    /// Equity at each valued tick and settlement.
    pub equity_curve: Vec<(u64, Decimal)>,
    /// All executions.
    pub trades: Vec<ContractFill>,
    /// Maximum drawdown.
    pub max_drawdown: Decimal,
    /// Sharpe ratio approximation, if enough data.
    pub sharpe_ratio: Option<Decimal>,
    /// Accounting status of the run.
    pub accounting: AccountingStatus,
    /// One line per contract, in the order registered.
    pub contracts: Vec<ContractSummary>,
    /// Ticks with no valuation, as an open contract had no mark yet.
    pub unvalued_ticks: u64,
    /// Open contracts with no mark when the run ended.
    pub never_valued: Vec<InstrumentId>,
    /// The first equity point resting on the mark of an expired contract with no
    /// settlement.
    pub stale_from_ms: Option<u64>,
    /// Largest shortfall per asset from booking fills the balance did not cover.
    pub overdrafts: BTreeMap<Asset, Decimal>,
    /// Commands discarded: those returned after the data ended, which nothing could carry
    /// out, and those past [`MAX_COMMANDS_PER_INSTANT`].
    pub discarded_commands: u64,
    /// Total faults, the first of which [`ContractEngine::faults`] keeps. From the first
    /// one on, every new order was refused.
    pub fault_count: u64,
}

impl ContractRunResult {
    /// Whether the run started, booked and valued all it should, with no fault and no
    /// contract left past its expiry without a settlement, open with no mark, with a settlement the
    /// ledger refused, or with a funding instant of no declared rate.
    #[must_use]
    pub fn is_consistent(&self) -> bool {
        self.accounting.is_consistent()
            && self.fault_count == 0
            && self.never_valued.is_empty()
            && self
                .contracts
                .iter()
                .all(|c| !c.expired_unsettled && c.funding_missing == 0)
    }
}

/// The ledger's identifier of a funding payment, the same for an instrument and an instant
/// in any run. An instrument name may be as long as an identifier can be, so it enters as
/// its 128 bit FNV-1a hash. A collision would be refused by the ledger as a fault, and no
/// payment would be lost silently.
fn funding_id(instrument: &str, at_ms: u64) -> String {
    let hash = instrument.bytes().fold(
        0x6c62_272e_07bb_0142_62b8_2175_6295_c58d_u128,
        |hash, byte| {
            (hash ^ u128::from(byte)).wrapping_mul(0x0000_0000_0100_0000_0000_0000_0000_013b)
        },
    );
    format!("funding-{hash:032x}-{at_ms}")
}

/// What kind of contract a lane is.
#[derive(Debug)]
enum LaneTerms {
    Future(FutureTerms),
    Perpetual(PerpetualTerms),
}

impl LaneTerms {
    fn spec(&self) -> &financial_types::instrument::InstrumentSpec {
        match self {
            Self::Future(terms) => terms.spec(),
            Self::Perpetual(terms) => terms.spec(),
        }
    }
}

#[derive(Debug)]
struct Lane<D, F> {
    terms: LaneTerms,
    ledger_id: margin_rs::InstrumentId,
    data: D,
    fill_model: F,
    orders: Vec<SimOrder>,
    mark: Option<(Decimal, u64)>,
    expired: bool,
    settled: bool,
    settlement_failed: bool,
    max_position: Decimal,
    fills: u64,
    ignored_ticks: u64,
    funding: Decimal,
}

impl<D, F> Lane<D, F> {
    fn id(&self) -> &InstrumentId {
        self.terms.spec().id()
    }
}

/// What a valuation gives, all of it exact, or none of it.
#[derive(Debug, Clone, Copy, Default)]
struct Figures {
    total_pnl: Decimal,
    realized: Decimal,
    funding: Decimal,
    unrealized: Decimal,
    fees: Decimal,
    net_pnl: Decimal,
    equity: Option<Decimal>,
}

#[derive(Debug, Clone)]
enum PendingAction {
    Activate(usize, SimOrder),
    Cancel(SimOrderId),
}

#[derive(Debug, Clone)]
enum Payload {
    Report(ContractReport, Option<(Dir, ContractFill)>),
    Settlement(usize, SettlementPrice, Decimal),
}

#[derive(Debug, Clone)]
struct Delivery {
    at: u64,
    payload: Payload,
}

/// The engine: see the [module documentation](self).
#[derive(Debug)]
pub struct ContractEngine<S: ContractStrategy, D: HistoricalDataSource, F: FillModel> {
    config: ContractConfig,
    strategy: S,
    initial: Ledger,
    ledger: Ledger,
    settlement_asset: Option<Asset>,
    lanes: Vec<Lane<D, F>>,
    declared: Vec<(usize, SettlementPrice)>,
    expiries: BTreeMap<(u64, usize), ()>,
    settlements: BTreeMap<(u64, usize), SettlementPrice>,
    /// The funding still to run: its mark and rate.
    funding_due: BTreeMap<(u64, usize), (Decimal, Decimal)>,
    fundings: Vec<ContractFunding>,
    clock: SimClock,
    latency: SimLatencyConfig,
    jitter: JitterRng,
    next_message: u64,
    next_order_id: SimOrderId,
    next_execution_id: u64,
    execution_offset: u64,
    /// Each order's lane and tag.
    owners: HashMap<SimOrderId, (usize, u64)>,
    tags: HashMap<u64, SimOrderId>,
    /// Orders that ended, kept to answer a cancel that crosses their end.
    closed: HashMap<SimOrderId, SimOrder>,
    pending: BTreeMap<(u64, u64), PendingAction>,
    pending_activations: HashMap<SimOrderId, (u64, u64)>,
    cancel_targets: HashSet<SimOrderId>,
    awaiting_cancel: HashMap<SimOrderId, SimOrder>,
    deliveries: VecDeque<Delivery>,
    observed: VecDeque<(u64, usize, MarketTick)>,
    /// The time the strategy asked to be woken at.
    wake: Option<u64>,
    /// The last time it was woken at, so that asking for it again does not repeat it.
    woken_at: Option<u64>,
    view: ContractView,
    last_delivery_at: u64,
    last_observed_at: u64,
    policy: Option<Box<dyn PretradePolicy + Send + Sync + UnwindSafe + RefUnwindSafe>>,
    pretrade_log: Vec<PretradeRecord>,
    reports: Vec<ContractReport>,
    trades: Vec<ContractFill>,
    num_trades: u64,
    equity_curve: Vec<(u64, Decimal)>,
    peak_equity: Decimal,
    max_drawdown: Decimal,
    /// Realized PnL and fees the ledger held at the start.
    base_realized: Decimal,
    base_fees: Decimal,
    /// Funding the ledger held at the start.
    base_funding: Decimal,
    figures: Figures,
    unvalued_ticks: u64,
    stale_from_ms: Option<u64>,
    overdrafts: BTreeMap<Asset, Decimal>,
    accounting_errors: Vec<(u64, MMError)>,
    valuation_failures: u64,
    dropped_fills: u64,
    setup_refused: Option<String>,
    faults: Vec<MMError>,
    fault_count: u64,
    data_ended: bool,
    discarded_commands: u64,
    /// The time of the commands being counted, and how many.
    commands_now: (u64, u64),
    /// A run started, and no reset came after it.
    started: bool,
}

impl<S: ContractStrategy, D: HistoricalDataSource, F: FillModel> ContractEngine<S, D, F> {
    /// An engine over `ledger`, whose current state a reset restores. Add contracts
    /// with [`ContractEngine::with_contract`].
    #[must_use]
    pub fn new(config: ContractConfig, ledger: Ledger, strategy: S) -> Self {
        let execution_offset = largest_execution_number(&ledger);
        let peak_equity = config.initial_capital;
        Self {
            config,
            strategy,
            initial: ledger.clone(),
            ledger,
            settlement_asset: None,
            lanes: Vec::new(),
            declared: Vec::new(),
            expiries: BTreeMap::new(),
            settlements: BTreeMap::new(),
            funding_due: BTreeMap::new(),
            fundings: Vec::new(),
            clock: SimClock::new(),
            latency: SimLatencyConfig::default(),
            jitter: JitterRng::new(0),
            next_message: 0,
            next_order_id: 1,
            next_execution_id: 1,
            execution_offset,
            owners: HashMap::new(),
            tags: HashMap::new(),
            closed: HashMap::new(),
            pending: BTreeMap::new(),
            pending_activations: HashMap::new(),
            cancel_targets: HashSet::new(),
            awaiting_cancel: HashMap::new(),
            deliveries: VecDeque::new(),
            observed: VecDeque::new(),
            wake: None,
            woken_at: None,
            view: ContractView::default(),
            last_delivery_at: 0,
            last_observed_at: 0,
            policy: None,
            pretrade_log: Vec::new(),
            reports: Vec::new(),
            trades: Vec::new(),
            num_trades: 0,
            equity_curve: Vec::new(),
            peak_equity,
            max_drawdown: Decimal::ZERO,
            base_realized: Decimal::ZERO,
            base_fees: Decimal::ZERO,
            base_funding: Decimal::ZERO,
            figures: Figures::default(),
            unvalued_ticks: 0,
            stale_from_ms: None,
            overdrafts: BTreeMap::new(),
            accounting_errors: Vec::new(),
            valuation_failures: 0,
            dropped_fills: 0,
            setup_refused: None,
            faults: Vec::new(),
            fault_count: 0,
            data_ended: false,
            discarded_commands: 0,
            commands_now: (0, 0),
            started: false,
        }
    }

    /// Adds a contract, with its data and its own fill model: one shared or cloned
    /// would correlate the legs, by its queue state or its random seed.
    ///
    /// # Errors
    ///
    /// [`MMError::InvalidConfiguration`] for a contract already added, not registered
    /// in the ledger or of other terms there, in another settlement asset than the
    /// others, or a fee scale above that asset's; and the error of
    /// [`FillModel::validate`].
    pub fn with_contract(self, terms: FutureTerms, data: D, fill_model: F) -> MMResult<Self> {
        self.add_lane(LaneTerms::Future(terms), data, fill_model)
    }

    /// Adds a perpetual with its declared funding, its data and its own fill model: see the
    /// [module documentation](self#funding).
    ///
    /// # Errors
    ///
    /// As [`ContractEngine::with_contract`].
    pub fn with_perpetual(self, terms: PerpetualTerms, data: D, fill_model: F) -> MMResult<Self> {
        self.add_lane(LaneTerms::Perpetual(terms), data, fill_model)
    }

    fn add_lane(mut self, terms: LaneTerms, data: D, fill_model: F) -> MMResult<Self> {
        fill_model.validate()?;
        let name = terms.spec().id().as_str().to_string();
        let refuse = |reason: String| {
            Err(MMError::InvalidConfiguration(format!(
                "contract {name}: {reason}"
            )))
        };
        if self.lanes.iter().any(|lane| lane.id() == terms.spec().id()) {
            return refuse("already added".to_string());
        }
        let ledger_id = match identify_spec_in_ledger(terms.spec(), &self.ledger) {
            Ok(id) => id,
            Err(error) => return refuse(error.to_string()),
        };
        let Some(registered) = self.ledger.instrument(&ledger_id) else {
            return refuse("not registered in the ledger".to_string());
        };
        let asset = registered.quote().clone();
        if self
            .settlement_asset
            .as_ref()
            .is_some_and(|known| *known != asset)
        {
            return refuse(format!("settles in {}, unlike the others", asset.as_str()));
        }
        let scale = self.ledger.snapshot().assets.get(&asset).copied();
        if scale.is_none_or(|scale| self.config.fee_scale > scale) {
            return refuse(format!(
                "fee scale {} above the settlement asset's {scale:?}",
                self.config.fee_scale
            ));
        }
        self.settlement_asset = Some(asset);
        self.lanes.push(Lane {
            terms,
            ledger_id,
            data,
            fill_model,
            orders: Vec::new(),
            mark: None,
            expired: false,
            settled: false,
            settlement_failed: false,
            max_position: Decimal::ZERO,
            fills: 0,
            ignored_ticks: 0,
            funding: Decimal::ZERO,
        });
        self.schedule_contract_events();
        Ok(self)
    }

    /// Declares the settlement of a contract already added, which books when published.
    ///
    /// # Errors
    ///
    /// [`MMError::InvalidConfiguration`] for a contract not added or a perpetual, a price
    /// the contract refuses, or a second price for it. What the ledger refuses depends on the position
    /// at publication, so it shows then: [`ContractSummary::settlement_failed`] and a fault.
    pub fn with_settlement(mut self, price: SettlementPrice) -> MMResult<Self> {
        let name = price.instrument().as_str().to_string();
        let refuse = |reason: String| {
            Err(MMError::InvalidConfiguration(format!(
                "settlement of {name}: {reason}"
            )))
        };
        let Some(index) = self
            .lanes
            .iter()
            .position(|lane| lane.id().as_str() == name)
        else {
            return refuse("the contract was not added".to_string());
        };
        if self.declared.iter().any(|(known, _)| *known == index) {
            return refuse("already declared".to_string());
        }
        match self.lanes.get(index).map(|lane| &lane.terms) {
            Some(LaneTerms::Future(terms)) => {
                if let Err(error) = terms.check_settlement(&price) {
                    return refuse(error.to_string());
                }
            }
            _ => return refuse("a perpetual does not settle".to_string()),
        }
        self.declared.push((index, price));
        self.schedule_contract_events();
        Ok(self)
    }

    /// Passes every new order through `policy`, with the contract in the request and the
    /// quantity in contracts.
    ///
    /// # Errors
    ///
    /// [`MMError::InvalidConfiguration`] unless the policy is
    /// [`PretradePolicy::in_contracts`].
    pub fn with_pretrade_policy(
        mut self,
        policy: impl PretradePolicy + Send + Sync + UnwindSafe + RefUnwindSafe + 'static,
    ) -> MMResult<Self> {
        if !policy.in_contracts() {
            return Err(MMError::InvalidConfiguration(format!(
                "policy {} does not decide in contracts",
                policy.description()
            )));
        }
        self.policy = Some(Box::new(policy));
        Ok(self)
    }

    /// Applies simulated latency, the same as in the single contract engine.
    ///
    /// # Errors
    ///
    /// As [`BacktestEngine::with_latency`](super::BacktestEngine::with_latency).
    pub fn with_latency(mut self, latency: SimLatencyConfig) -> MMResult<Self> {
        latency.validate()?;
        self.latency = latency;
        self.jitter = JitterRng::new(latency.seed);
        Ok(self)
    }

    fn schedule_contract_events(&mut self) {
        self.expiries = self
            .lanes
            .iter()
            .enumerate()
            .filter_map(|(index, lane)| match &lane.terms {
                LaneTerms::Future(terms) => Some(((terms.expiry_ms(), index), ())),
                LaneTerms::Perpetual(_) => None,
            })
            .collect();
        self.funding_due = self
            .lanes
            .iter()
            .enumerate()
            .filter_map(|(index, lane)| match &lane.terms {
                LaneTerms::Perpetual(terms) => Some((index, terms)),
                LaneTerms::Future(_) => None,
            })
            .flat_map(|(index, terms)| {
                terms
                    .points()
                    .iter()
                    .map(move |point| ((point.at_ms, index), (point.mark, point.rate)))
            })
            .collect();
        self.settlements = self
            .declared
            .iter()
            .map(|(index, price)| ((price.published_ms(), *index), price.clone()))
            .collect();
    }

    /// The ledger, the truth of what booked.
    #[must_use]
    pub fn ledger(&self) -> &Ledger {
        &self.ledger
    }

    /// The strategy.
    #[must_use]
    pub fn strategy(&self) -> &S {
        &self.strategy
    }

    /// The strategy, to change it.
    pub fn strategy_mut(&mut self) -> &mut S {
        &mut self.strategy
    }

    /// The reports recorded so far, in event order.
    #[must_use]
    pub fn execution_reports(&self) -> &[ContractReport] {
        &self.reports
    }

    /// Every pretrade decision taken so far, in decision order.
    #[must_use]
    pub fn pretrade_decisions(&self) -> &[PretradeRecord] {
        &self.pretrade_log
    }

    /// The first errors of a component, in order. From the first one on, every new order
    /// is refused until [`ContractEngine::reset`]; orders already resting keep working.
    #[must_use]
    pub fn faults(&self) -> &[MMError] {
        &self.faults
    }

    /// Total faults since the last reset, recorded or not. Saturates at `u64::MAX`.
    #[must_use]
    pub fn fault_count(&self) -> u64 {
        self.fault_count
    }

    /// Bookings the ledger refused, as `(execution number of this run, error)`.
    #[must_use]
    pub fn accounting_errors(&self) -> &[(u64, MMError)] {
        &self.accounting_errors
    }

    /// Resets for another run, restoring the ledger, the data, the models, the
    /// strategy, the policy and every counter.
    pub fn reset(&mut self) {
        self.ledger = self.initial.clone();
        for lane in &mut self.lanes {
            lane.data.reset();
            lane.fill_model.reset();
            lane.orders.clear();
            lane.mark = None;
            lane.expired = false;
            lane.settled = false;
            lane.settlement_failed = false;
            lane.max_position = Decimal::ZERO;
            lane.fills = 0;
            lane.ignored_ticks = 0;
            lane.funding = Decimal::ZERO;
        }
        self.fundings.clear();
        self.schedule_contract_events();
        self.strategy.reset();
        if let Some(policy) = self.policy.as_mut() {
            policy.reset();
        }
        self.clock.reset();
        self.jitter.reset();
        self.next_message = 0;
        self.next_order_id = 1;
        self.next_execution_id = 1;
        self.owners.clear();
        self.tags.clear();
        self.closed.clear();
        self.pending.clear();
        self.pending_activations.clear();
        self.cancel_targets.clear();
        self.awaiting_cancel.clear();
        self.deliveries.clear();
        self.observed.clear();
        self.woken_at = None;
        self.view = ContractView::default();
        self.last_delivery_at = 0;
        self.last_observed_at = 0;
        self.pretrade_log.clear();
        self.reports.clear();
        self.trades.clear();
        self.num_trades = 0;
        self.equity_curve.clear();
        self.peak_equity = self.config.initial_capital;
        self.max_drawdown = Decimal::ZERO;
        self.figures = Figures::default();
        self.unvalued_ticks = 0;
        self.stale_from_ms = None;
        self.overdrafts.clear();
        self.accounting_errors.clear();
        self.valuation_failures = 0;
        self.dropped_fills = 0;
        self.setup_refused = None;
        self.faults.clear();
        self.fault_count = 0;
        self.data_ended = false;
        self.discarded_commands = 0;
        self.commands_now = (0, 0);
        self.started = false;
    }

    /// Refuses what cannot be valued, and fixes where the run starts.
    fn start(&mut self) -> MMResult<()> {
        if self.started {
            return Err(MMError::InvalidConfiguration(
                "a contract run needs a reset before running again".to_string(),
            ));
        }
        if self.lanes.is_empty() {
            return Err(MMError::InvalidConfiguration(
                "a contract run needs a contract".to_string(),
            ));
        }
        let snapshot = self.ledger.snapshot();
        if let Some((other, _)) = snapshot.positions.iter().find(|(id, position)| {
            !position.is_flat() && !self.lanes.iter().any(|lane| lane.ledger_id == **id)
        }) {
            return Err(MMError::InvalidConfiguration(format!(
                "{} is open in the ledger and has no data, so it cannot be valued",
                other.as_str()
            )));
        }
        let Some(asset) = self.settlement_asset.clone() else {
            return Err(MMError::InvalidConfiguration(
                "a contract run needs a settlement asset".to_string(),
            ));
        };
        // As `Ledger::valuation` sums them, with what no mark changes.
        let held = snapshot
            .positions
            .iter()
            .filter(|(id, _)| {
                self.ledger
                    .instrument(id)
                    .is_some_and(|instrument| *instrument.quote() == asset)
            })
            .map(|(_, position)| position.realized_pnl);
        let settled = snapshot.settled_pnl.get(&asset).copied();
        let funding = snapshot.funding.get(&asset).copied();
        self.base_realized = held
            .chain(settled)
            .chain(funding)
            .try_fold(Decimal::ZERO, |sum, value| sum.checked_add(value))
            .or_overflow("ledger realized pnl")?;
        self.base_fees = self.ledger.fees_paid(&asset);
        self.base_funding = self.ledger.funding(&asset);
        self.started = true;
        let ledger = &self.ledger;
        self.view.positions = self
            .lanes
            .iter_mut()
            .map(|lane| {
                let position = ledger
                    .position(&lane.ledger_id)
                    .map_or(Decimal::ZERO, |p| p.quantity);
                lane.max_position = lane.max_position.max(position.abs());
                (lane.id().clone(), position)
            })
            .collect();
        self.ask_wake();
        Ok(())
    }

    /// The next tick by time, then by registration.
    fn next_tick(&mut self) -> Option<(usize, MarketTick)> {
        let (index, _) = self
            .lanes
            .iter()
            .enumerate()
            .filter_map(|(index, lane)| lane.data.peek_tick().map(|tick| (index, tick.timestamp)))
            .min_by_key(|(index, time)| (*time, *index))?;
        let tick = self.lanes.get_mut(index)?.data.next_tick()?;
        Some((index, tick))
    }

    /// Runs the backtest: see the [module documentation](self). A second run needs a
    /// [`ContractEngine::reset`] first, or it is refused, with nothing processed.
    pub fn run(&mut self) -> ContractRunResult {
        let (mut num_ticks, mut start_time, mut end_time) = (0u64, 0u64, 0u64);
        self.setup_refused = None;
        if let Err(error) = self.start() {
            self.setup_refused = Some(error.to_string());
            self.record_fault(error);
            return self.result(num_ticks, start_time, end_time);
        }
        while let Some((index, tick)) = self.next_tick() {
            if num_ticks == 0 {
                start_time = tick.timestamp;
                self.funding_due.retain(|(at_ms, _), _| *at_ms > start_time);
            }
            end_time = tick.timestamp;
            num_ticks = num_ticks.saturating_add(1);

            let now = self.clock.now().max(tick.timestamp);
            self.advance_events(now);
            let _ = self.clock.advance_to(now);

            let Some(lane) = self.lanes.get_mut(index) else {
                continue;
            };
            if lane.expired {
                lane.ignored_ticks = lane.ignored_ticks.saturating_add(1);
                continue;
            }
            if let Some(policy) = self.policy.as_mut() {
                policy.on_market_data(tick.mid_price(), tick.timestamp);
            }
            self.match_active_orders(index, &tick, now);

            let ready = self
                .delayed(tick.timestamp, self.latency.observation_ms)
                .max(self.last_observed_at);
            self.last_observed_at = ready;
            self.observed.push_back((ready, index, tick.clone()));
            self.advance_events(now);

            if let Some(lane) = self.lanes.get_mut(index) {
                lane.mark = Some((tick.mid_price(), tick.timestamp));
            }
            self.sample(tick.timestamp);
        }
        if num_ticks == 0 {
            // No first tick, so no instant is after it.
            self.funding_due.clear();
        }
        self.finish();
        self.result(num_ticks, start_time, end_time)
    }

    /// With no data left: settlements still declared book, and reports still on their way
    /// are delivered.
    fn finish(&mut self) {
        self.data_ended = true;
        self.observed.clear();
        if let Some((published, _)) = self.settlements.keys().next_back().copied() {
            self.advance_events(published);
        }
        while !self.deliveries.is_empty() {
            self.deliver_next();
        }
    }

    /// Runs every scheduled event at or before `until`, in time order.
    fn advance_events(&mut self, until: u64) {
        loop {
            let times = [
                self.funding_due.keys().next().map(|key| key.0),
                self.expiries.keys().next().map(|key| key.0),
                self.pending.keys().next().map(|key| key.0),
                self.settlements.keys().next().map(|key| key.0),
                self.deliveries.front().map(|d| d.at),
                self.wake,
                self.observed.front().map(|(at, _, _)| *at),
            ];
            let Some((time, class)) = next_due(&times, until) else {
                break;
            };
            let at = time.max(self.clock.now());
            let _ = self.clock.advance_to(at);
            match class {
                0 => self.fund_next(),
                1 => self.expire_next(),
                2 => self.apply_next_venue_message(),
                3 => self.settle_next(),
                4 => self.deliver_next(),
                5 => self.wake_strategy(at),
                _ => self.observe_next(at),
            }
        }
    }

    /// The position pays or receives one funding instant, through the ledger.
    fn fund_next(&mut self) {
        let Some(((at_ms, index), (mark, rate))) = self.funding_due.pop_first() else {
            return;
        };
        let Some(lane) = self.lanes.get(index) else {
            return;
        };
        let position = self
            .ledger
            .position(&lane.ledger_id)
            .map_or(Decimal::ZERO, |p| p.quantity);
        if position.is_zero() {
            return;
        }
        match self.book_funding(index, at_ms, position, mark, rate) {
            Ok(amount) => {
                if self.config.record_trades
                    && let Some(lane) = self.lanes.get(index)
                {
                    self.fundings.push(ContractFunding {
                        contract: lane.id().clone(),
                        at_ms,
                        position,
                        mark,
                        rate,
                        amount,
                    });
                }
            }
            Err(error) => self.record_fault(error),
        }
        self.sample(at_ms);
    }

    fn book_funding(
        &mut self,
        index: usize,
        at_ms: u64,
        position: Decimal,
        mark: Decimal,
        rate: Decimal,
    ) -> MMResult<Decimal> {
        let error = |context: &str, error: margin_rs::MarginError| {
            MMError::InvalidPositionUpdate(format!("margin ledger funding {context}: {error}"))
        };
        let asset = self
            .settlement_asset
            .clone()
            .or_overflow("settlement asset")?;
        let scale = self
            .ledger
            .snapshot()
            .assets
            .get(&asset)
            .copied()
            .or_overflow("settlement scale")?;
        let lane = self.lanes.get_mut(index).or_overflow("funding lane")?;
        let paid = settlement_charge(lane.terms.spec(), scale, mark, position.abs(), rate)?.amount;
        let amount = if position > Decimal::ZERO {
            Decimal::ZERO.checked_sub(paid).or_overflow("funding")?
        } else {
            paid
        };
        if amount.is_zero() {
            return Ok(amount);
        }
        let id = funding_id(lane.ledger_id.as_str(), at_ms);
        let booked = self
            .ledger
            .book_funding(&FundingPayment {
                id: ExecutionId::new(id.clone()).map_err(|e| error("id", e))?,
                instrument: lane.ledger_id.clone(),
                amount,
                as_of_ms: at_ms,
            })
            .map_err(|e| error("booking", e))?;
        if !booked.is_new {
            return Err(MMError::InvalidPositionUpdate(format!(
                "margin ledger funding booking: {id} was already booked"
            )));
        }
        lane.funding = lane.funding.checked_add(amount).or_overflow("funding")?;
        if let Some(shortfall) = booked.shortfall {
            let worst = self.overdrafts.entry(asset).or_insert(Decimal::ZERO);
            *worst = (*worst).max(shortfall);
        }
        Ok(amount)
    }

    /// The venue closes a contract at its expiry.
    fn expire_next(&mut self) {
        let Some(((_, index), ())) = self.expiries.pop_first() else {
            return;
        };
        let Some(lane) = self.lanes.get_mut(index) else {
            return;
        };
        lane.expired = true;
        let mut ending = std::mem::take(&mut lane.orders);
        let on_their_way: Vec<(u64, u64)> = self
            .pending
            .iter()
            .filter_map(|(key, action)| match action {
                PendingAction::Activate(lane, _) if *lane == index => Some(*key),
                _ => None,
            })
            .collect();
        for key in on_their_way {
            if let Some(PendingAction::Activate(_, order)) = self.pending.remove(&key) {
                self.pending_activations.remove(&order.id);
                ending.push(order);
            }
        }
        ending.sort_unstable_by_key(|order| order.id);
        for mut order in ending {
            let stamp = self.clock.stamp();
            let _ = order.cancel();
            let report = ExecutionReport::status(stamp, &order, ExecutionKind::Canceled);
            self.retire(order);
            self.emit(report);
        }
    }

    /// Keeps an order that ended, to answer a cancel that crosses its end.
    fn retire(&mut self, order: SimOrder) {
        if self.cancel_targets.contains(&order.id) {
            self.awaiting_cancel.insert(order.id, order.clone());
        }
        self.closed.insert(order.id, order);
    }

    fn settle_next(&mut self) {
        let Some(((published, index), price)) = self.settlements.pop_first() else {
            return;
        };
        let Some(lane) = self.lanes.get_mut(index) else {
            return;
        };
        let before = self
            .ledger
            .position(&lane.ledger_id)
            .map_or(Decimal::ZERO, |p| p.quantity);
        let LaneTerms::Future(terms) = &lane.terms else {
            return;
        };
        match terms.settle_in_ledger(&mut self.ledger, &price) {
            Ok(_) => lane.settled = true,
            Err(error) => {
                lane.settlement_failed = true;
                self.record_fault(MMError::InvalidPositionUpdate(format!(
                    "settlement of {}: {error}",
                    price.instrument().as_str()
                )));
                return;
            }
        }
        let at = self
            .delayed(published, self.latency.ack_ms)
            .max(self.last_delivery_at);
        self.last_delivery_at = at;
        self.deliveries.push_back(Delivery {
            at,
            payload: Payload::Settlement(index, price, before),
        });
        self.sample(published);
    }

    fn observe_next(&mut self, at: u64) {
        let Some((_, index, tick)) = self.observed.pop_front() else {
            return;
        };
        let Some(lane) = self.lanes.get(index) else {
            return;
        };
        let contract = lane.id().clone();
        self.view.now_ms = at;
        let commands = self.strategy.on_tick(&contract, &tick, &self.view);
        self.after_callback(commands, at);
    }

    fn match_active_orders(&mut self, index: usize, tick: &MarketTick, now: u64) {
        let Some(lane) = self.lanes.get_mut(index) else {
            return;
        };
        lane.fill_model.on_tick(tick);
        let count = lane.orders.len();
        for position in 0..count {
            let Some(lane) = self.lanes.get(index) else {
                return;
            };
            let Some(order) = lane.orders.get(position) else {
                continue;
            };
            if !order.is_eligible_at(tick.timestamp) {
                continue;
            }
            let side = order.side;
            let dir = match Dir::try_from(side) {
                Ok(dir) => dir,
                Err(error) => {
                    self.record_fault(error);
                    continue;
                }
            };
            let view = SimulatedOrder::new(side, order.price, order.quantity, order.active_after);
            let time_in_queue = tick.timestamp.saturating_sub(order.active_after);
            let outcome = match lane.fill_model.simulate_fill(&view, tick, time_in_queue) {
                Ok(outcome) => outcome,
                Err(error) => {
                    self.record_fault(error);
                    continue;
                }
            };
            let Some((quantity, model_price)) = fill_increment(order, outcome) else {
                continue;
            };
            if quantity <= Decimal::ZERO {
                continue;
            }
            let price = slipped_price(&self.config.slippage, model_price, dir);
            if price <= Decimal::ZERO {
                continue;
            }
            let charge = match settlement_charge(
                lane.terms.spec(),
                self.config.fee_scale,
                price,
                quantity,
                self.config.fee_rate,
            ) {
                Ok(charge) => charge,
                Err(error) => {
                    self.dropped_fills = self.dropped_fills.saturating_add(1);
                    self.record_fault(error);
                    continue;
                }
            };
            self.execute(index, position, dir, price, quantity, &charge, now);
        }
        let Some(lane) = self.lanes.get_mut(index) else {
            return;
        };
        let orders = std::mem::take(&mut lane.orders);
        let (ended, live): (Vec<SimOrder>, Vec<SimOrder>) =
            orders.into_iter().partition(|o| o.state.is_terminal());
        lane.orders = live;
        for order in ended {
            self.retire(order);
        }
    }

    /// One execution: the policy, then the ledger, then the report channel.
    #[allow(clippy::too_many_arguments)]
    fn execute(
        &mut self,
        index: usize,
        position: usize,
        dir: Dir,
        price: Decimal,
        quantity: Decimal,
        charge: &FeeCharge,
        now: u64,
    ) {
        // Before anything moves: a number used up must not name an execution again.
        let execution = self.next_execution_id;
        let numbers = execution
            .checked_add(1)
            .zip(self.execution_offset.checked_add(execution));
        let Some((next_execution_id, number)) = numbers else {
            self.dropped_fills = self.dropped_fills.saturating_add(1);
            self.record_fault(MMError::NumericalError(
                "execution numbers are used up".to_string(),
            ));
            return;
        };
        let Some(lane) = self.lanes.get_mut(index) else {
            return;
        };
        let contract = lane.id().clone();
        let Some(order) = lane.orders.get_mut(position) else {
            return;
        };
        if order.apply_fill(quantity).is_err() {
            return;
        }
        let stamp = self.clock.stamp();
        let report = ExecutionReport::fill(stamp, order, price, quantity, charge.quote_value);
        let (order_id, cumulative, side) = (order.id, order.filled, order.side);
        let tag = self.owners.get(&order_id).map_or(0, |(_, tag)| *tag);
        self.next_execution_id = next_execution_id;
        let execution_id = format!("sim-{number}");
        self.notify_policy(OrderEvent::Fill {
            order_id: order_id.to_string(),
            execution_id: execution_id.clone(),
            side,
            price,
            quantity,
            cumulative_quantity: cumulative,
            fee: charge.amount,
            fee_asset: FeeAsset::Settlement,
            timestamp: now,
        });
        let fill = ContractFill {
            contract: contract.clone(),
            tag,
            execution_id,
            side,
            price,
            quantity,
            cumulative_quantity: cumulative,
            fee: charge.amount,
            timestamp: now,
        };
        if let Err(error) = self.book(index, &fill, dir) {
            self.accounting_errors.push((execution, error));
        }
        self.num_trades = self.num_trades.saturating_add(1);
        if self.config.record_trades {
            self.trades.push(fill.clone());
        }
        self.queue_delivery(
            ContractReport {
                contract,
                tag,
                report,
            },
            Some((dir, fill)),
        );
    }

    /// Books a fill in the ledger, with overdraft, as it happened at the venue.
    fn book(&mut self, index: usize, fill: &ContractFill, dir: Dir) -> MMResult<()> {
        let error = |context: &str, error: margin_rs::MarginError| {
            MMError::InvalidPositionUpdate(format!("margin ledger {context}: {error}"))
        };
        let id = ExecutionId::new(fill.execution_id.clone()).map_err(|e| error("id", e))?;
        let Some(lane) = self.lanes.get_mut(index) else {
            return Ok(());
        };
        let mut execution = Execution::new(
            id,
            lane.ledger_id.clone(),
            margin_rs::Action::from(dir),
            fill.price,
            fill.quantity,
        )
        .with_executed_at(fill.timestamp);
        if !fill.fee.is_zero()
            && let Some(asset) = &self.settlement_asset
        {
            execution = execution.with_fee(asset.clone(), fill.fee);
        }
        let booked = self
            .ledger
            .book_execution_allowing_overdraft(&execution, None)
            .map_err(|e| error("booking", e))?;
        if !matches!(booked.booking, Booking::Booked(_)) {
            return Err(MMError::InvalidPositionUpdate(format!(
                "margin ledger booking: {} was already booked",
                fill.execution_id
            )));
        }
        lane.fills = lane.fills.saturating_add(1);
        for (asset, shortfall) in booked.shortfalls {
            let worst = self.overdrafts.entry(asset).or_insert(Decimal::ZERO);
            *worst = (*worst).max(shortfall);
        }
        let position = self
            .ledger
            .position(&lane.ledger_id)
            .map_or(Decimal::ZERO, |p| p.quantity);
        lane.max_position = lane.max_position.max(position.abs());
        Ok(())
    }

    fn apply_commands(&mut self, commands: Vec<ContractCommand>, now: u64) {
        let count = crate::types::decimal::saturating_u64(commands.len());
        if self.data_ended {
            self.discarded_commands = self.discarded_commands.saturating_add(count);
            return;
        }
        if self.commands_now.0 != now {
            self.commands_now = (now, 0);
        }
        let so_far = self.commands_now.1;
        self.commands_now.1 = so_far.saturating_add(count);
        if self.commands_now.1 > MAX_COMMANDS_PER_INSTANT {
            if so_far <= MAX_COMMANDS_PER_INSTANT {
                self.record_fault(MMError::InvalidMarketState(format!(
                    "more than {MAX_COMMANDS_PER_INSTANT} commands at {now}"
                )));
            }
            self.discarded_commands = self.discarded_commands.saturating_add(count);
            return;
        }
        for command in commands {
            match command {
                ContractCommand::Submit(order) => self.submit(order, now),
                ContractCommand::Cancel { tag } => self.cancel_tag(tag, now),
                ContractCommand::CancelAll { contract } => self.cancel_all(&contract, now),
            }
        }
    }

    fn lane_of(&self, contract: &InstrumentId) -> Option<usize> {
        self.lanes.iter().position(|lane| lane.id() == contract)
    }

    fn submit(&mut self, request: ContractOrder, now: u64) {
        let ContractOrder {
            tag,
            contract,
            side,
            price,
            quantity,
            purpose,
        } = request;
        let Some(index) = self.lane_of(&contract) else {
            self.record_fault(MMError::InvalidConfiguration(format!(
                "order {tag} on {}, which is not in this run",
                contract.as_str()
            )));
            return;
        };
        if self.tags.contains_key(&tag) {
            self.record_fault(MMError::InvalidConfiguration(format!(
                "order tag {tag} was already used"
            )));
            return;
        }
        let id = self.next_order_id;
        let Some(next_order_id) = id.checked_add(1) else {
            self.record_fault(MMError::NumericalError(
                "order identifiers are used up".to_string(),
            ));
            return;
        };
        self.next_order_id = next_order_id;
        self.tags.insert(tag, id);
        self.owners.insert(id, (index, tag));
        let stamp = self.clock.stamp();
        let Some(lane) = self.lanes.get(index) else {
            return;
        };
        let spec = lane.terms.spec().clone();
        let mut size = quantity;
        let mut refused = !self.faults.is_empty()
            || lane.expired
            || Dir::try_from(side).is_err()
            || spec.check_price(price).is_err()
            || spec.check_lot(quantity).is_err();
        if !refused && let Some(policy) = self.policy.as_mut() {
            let mut pretrade =
                PretradeRequest::new(id.to_string(), side, Some(price), size, purpose, now);
            pretrade.instrument = Some(contract.as_str().to_string());
            let decision = policy.check(&pretrade);
            let approved = decision.approved_quantity(size);
            let whole_lots = spec.check_lot(approved).is_ok();
            refused = approved <= Decimal::ZERO || !whole_lots;
            if approved > Decimal::ZERO
                && !whole_lots
                && let Err(error) = policy.on_order_event(&OrderEvent::VenueRejected {
                    order_id: id.to_string(),
                })
            {
                self.record_fault(error);
            }
            if !refused {
                size = approved;
            }
            self.pretrade_log.push(PretradeRecord {
                order_id: id,
                request: pretrade,
                decision,
                submitted: if refused { Decimal::ZERO } else { size },
            });
        }
        if refused {
            let mut order = SimOrder::new(id, side, price, size, stamp, now);
            let _ = order.reject();
            let stamp = self.clock.stamp();
            let report = ExecutionReport::status(stamp, &order, ExecutionKind::Rejected);
            self.closed.insert(id, order);
            self.queue_delivery(
                ContractReport {
                    contract,
                    tag,
                    report,
                },
                None,
            );
            return;
        }
        let effective = self.delayed(now, self.latency.submit_ms);
        let order = SimOrder::new(id, side, price, size, stamp, effective);
        self.schedule(effective, PendingAction::Activate(index, order));
    }

    fn cancel_tag(&mut self, tag: u64, now: u64) {
        let Some(id) = self.tags.get(&tag).copied() else {
            self.record_fault(MMError::InvalidConfiguration(format!(
                "cancel of order tag {tag}, which was never sent"
            )));
            return;
        };
        if self.cancel_targets.contains(&id) {
            return;
        }
        if let Some(order) = self.closed.get(&id).cloned() {
            let stamp = self.clock.stamp();
            let report = ExecutionReport::status(stamp, &order, ExecutionKind::CancelRejected);
            if let Some(report) = self.with_owner(report) {
                self.queue_delivery(report, None);
            }
            return;
        }
        self.request_cancel(id, now);
    }

    /// As the single contract engine replaces its quotes: in order of identifier, so
    /// the jitter draws are deterministic.
    fn cancel_all(&mut self, contract: &InstrumentId, now: u64) {
        let Some(index) = self.lane_of(contract) else {
            self.record_fault(MMError::InvalidConfiguration(format!(
                "cancel on {}, which is not in this run",
                contract.as_str()
            )));
            return;
        };
        let Some(lane) = self.lanes.get(index) else {
            return;
        };
        let mut targets: Vec<SimOrderId> = lane
            .orders
            .iter()
            .filter(|o| !o.state.is_terminal())
            .map(|o| o.id)
            .collect();
        targets.extend(
            self.pending_activations
                .keys()
                .filter(|id| self.owners.get(id).is_some_and(|(lane, _)| *lane == index)),
        );
        targets.retain(|id| !self.cancel_targets.contains(id));
        targets.sort_unstable();
        for id in targets {
            self.request_cancel(id, now);
        }
    }

    fn request_cancel(&mut self, id: SimOrderId, now: u64) {
        let effective = self.delayed(now, self.latency.cancel_ms);
        self.cancel_targets.insert(id);
        self.notify_policy(OrderEvent::CancelRequested {
            order_id: id.to_string(),
        });
        self.schedule(effective, PendingAction::Cancel(id));
    }

    fn delayed(&mut self, now: u64, base_ms: u64) -> u64 {
        let jitter = self.jitter.draw(self.latency.jitter_ms);
        now.saturating_add(base_ms).saturating_add(jitter)
    }

    fn schedule(&mut self, effective: u64, action: PendingAction) {
        let key = (effective, self.next_message);
        let Some(next_message) = self.next_message.checked_add(1) else {
            self.record_fault(MMError::NumericalError(
                "venue message numbers are used up".to_string(),
            ));
            return;
        };
        self.next_message = next_message;
        if let PendingAction::Activate(_, order) = &action {
            self.pending_activations.insert(order.id, key);
        }
        self.pending.insert(key, action);
    }

    fn apply_next_venue_message(&mut self) {
        let Some((_, action)) = self.pending.pop_first() else {
            return;
        };
        match action {
            PendingAction::Activate(index, mut order) => {
                self.pending_activations.remove(&order.id);
                let kind = if order.validate().is_ok() {
                    let _ = order.accept();
                    ExecutionKind::Accepted
                } else {
                    let _ = order.reject();
                    ExecutionKind::Rejected
                };
                let stamp = self.clock.stamp();
                let report = ExecutionReport::status(stamp, &order, kind);
                if kind == ExecutionKind::Accepted {
                    if let Some(lane) = self.lanes.get_mut(index) {
                        lane.orders.push(order);
                    }
                } else {
                    self.retire(order);
                }
                self.emit(report);
            }
            PendingAction::Cancel(id) => {
                self.cancel_targets.remove(&id);
                self.apply_cancel(id);
            }
        }
    }

    fn apply_cancel(&mut self, id: SimOrderId) {
        let stamp = self.clock.stamp();
        let lane = self.owners.get(&id).map(|(lane, _)| *lane);
        if let Some(lane) = lane.and_then(|lane| self.lanes.get_mut(lane))
            && let Some(index) = lane.orders.iter().position(|o| o.id == id)
            && let Some(order) = lane.orders.get_mut(index)
            && order.cancel().is_ok()
        {
            let report = ExecutionReport::status(stamp, order, ExecutionKind::Canceled);
            let order = lane.orders.remove(index);
            self.closed.insert(id, order);
            self.emit(report);
            return;
        }
        if let Some(key) = self.pending_activations.remove(&id) {
            if let Some(PendingAction::Activate(_, mut order)) = self.pending.remove(&key) {
                let _ = order.cancel();
                let report = ExecutionReport::status(stamp, &order, ExecutionKind::Canceled);
                self.closed.insert(id, order);
                self.emit(report);
            }
            return;
        }
        if let Some(order) = self.awaiting_cancel.remove(&id) {
            let report = ExecutionReport::status(stamp, &order, ExecutionKind::CancelRejected);
            self.emit(report);
        }
    }

    fn with_owner(&self, report: ExecutionReport) -> Option<ContractReport> {
        let (lane, tag) = self.owners.get(&report.order_id).copied()?;
        let contract = self.lanes.get(lane)?.id().clone();
        Some(ContractReport {
            contract,
            tag,
            report,
        })
    }

    fn emit(&mut self, report: ExecutionReport) {
        let order_id = report.order_id.to_string();
        let event = match report.kind {
            ExecutionKind::Accepted => Some(OrderEvent::Accepted { order_id }),
            ExecutionKind::Rejected => Some(OrderEvent::VenueRejected { order_id }),
            ExecutionKind::Canceled => Some(OrderEvent::Canceled { order_id }),
            ExecutionKind::CancelRejected => Some(OrderEvent::CancelRejected { order_id }),
            ExecutionKind::PartialFill | ExecutionKind::Fill => None,
        };
        if let Some(event) = event {
            self.notify_policy(event);
        }
        if let Some(report) = self.with_owner(report) {
            self.queue_delivery(report, None);
        }
    }

    fn queue_delivery(&mut self, report: ContractReport, fill: Option<(Dir, ContractFill)>) {
        let at = self
            .delayed(report.report.stamp.timestamp, self.latency.ack_ms)
            .max(self.last_delivery_at);
        self.last_delivery_at = at;
        if self.config.record_trades {
            self.reports.push(report.clone());
        }
        self.deliveries.push_back(Delivery {
            at,
            payload: Payload::Report(report, fill),
        });
    }

    fn deliver_next(&mut self) {
        let Some(delivery) = self.deliveries.pop_front() else {
            return;
        };
        let at = delivery.at.max(self.clock.now());
        self.view.now_ms = at;
        match delivery.payload {
            Payload::Report(report, fill) => {
                let commands = self.strategy.on_report(&report, &self.view);
                self.after_callback(commands, at);
                if let Some((dir, fill)) = fill {
                    self.see_fill(dir, &fill);
                    let commands = self.strategy.on_fill(&fill, &self.view);
                    self.after_callback(commands, at);
                }
            }
            Payload::Settlement(index, price, contracts) => {
                let Some(contract) = self.lanes.get(index).map(|lane| lane.id().clone()) else {
                    return;
                };
                if let Some((_, position)) = self
                    .view
                    .positions
                    .iter_mut()
                    .find(|(id, _)| *id == contract)
                {
                    *position = Decimal::ZERO;
                }
                let commands = self
                    .strategy
                    .on_settlement(&contract, &price, contracts, &self.view);
                self.after_callback(commands, at);
            }
        }
    }

    fn wake_strategy(&mut self, at: u64) {
        self.wake = None;
        self.woken_at = Some(at);
        self.view.now_ms = at;
        let commands = self.strategy.on_wake(&self.view);
        self.after_callback(commands, at);
    }

    /// What every callback leaves: its faults first, as they stop its commands.
    fn after_callback(&mut self, commands: Vec<ContractCommand>, at: u64) {
        for _ in 0..MAX_STRATEGY_FAULTS_PER_CALLBACK {
            let Some(error) = self.strategy.take_fault() else {
                break;
            };
            self.record_fault(error);
        }
        self.apply_commands(commands, at);
        self.ask_wake();
    }

    fn ask_wake(&mut self) {
        let now = self.clock.now();
        let woken_at = self.woken_at;
        self.wake = self
            .strategy
            .next_wake_ms()
            .filter(|wake| *wake >= now && woken_at != Some(*wake));
    }

    /// A delivered fill moves what the strategy knows.
    fn see_fill(&mut self, dir: Dir, fill: &ContractFill) {
        let moved = match dir {
            Dir::Buy => fill.quantity,
            Dir::Sell => Decimal::ZERO.saturating_sub(fill.quantity),
        };
        let Some((_, position)) = self
            .view
            .positions
            .iter_mut()
            .find(|(id, _)| *id == fill.contract)
        else {
            return;
        };
        match position.checked_add(moved) {
            Some(sum) => *position = sum,
            None => self.record_fault(MMError::NumericalError(
                "the position a strategy knows overflowed".to_string(),
            )),
        }
    }

    fn record_fault(&mut self, error: MMError) {
        self.fault_count = self.fault_count.saturating_add(1);
        if self.faults.len() < MAX_RECORDED_FAULTS {
            self.faults.push(error);
        }
    }

    fn notify_policy(&mut self, event: OrderEvent) {
        if let Some(policy) = self.policy.as_mut()
            && let Err(error) = policy.on_order_event(&event)
        {
            self.record_fault(error);
        }
    }

    /// Whether an open contract has no mark yet.
    fn lacks_mark(&self) -> bool {
        self.lanes.iter().any(|lane| {
            lane.mark.is_none()
                && self
                    .ledger
                    .position(&lane.ledger_id)
                    .is_some_and(|p| !p.is_flat())
        })
    }

    /// The figures of a valuation line, or `None` if one overflows.
    fn figures_of(&self, line: &AssetPnl) -> Option<Figures> {
        let realized = line.realized_pnl.checked_sub(self.base_realized)?;
        let funding = self
            .settlement_asset
            .as_ref()
            .map(|asset| self.ledger.funding(asset))?
            .checked_sub(self.base_funding)?;
        let fees = line.fees_paid.checked_sub(self.base_fees)?;
        let total_pnl = realized.checked_add(line.unrealized_pnl)?;
        let net_pnl = total_pnl.checked_sub(fees)?;
        Some(Figures {
            total_pnl,
            realized: realized.checked_sub(funding)?,
            funding,
            unrealized: line.unrealized_pnl,
            fees,
            net_pnl,
            equity: Some(self.config.initial_capital.checked_add(net_pnl)?),
        })
    }

    /// Reads the figures from the ledger at the latest marks. `false` if an open contract
    /// has no mark yet. A failure keeps the previous figures, and is a fault.
    fn value(&mut self) -> bool {
        if self.lacks_mark() {
            return false;
        }
        let marks: BTreeMap<_, _> = self
            .lanes
            .iter()
            .filter_map(|lane| lane.mark.map(|(mark, _)| (lane.ledger_id.clone(), mark)))
            .collect();
        let figures = self
            .ledger
            .valuation(&marks)
            .map_err(|error| {
                MMError::InvalidPositionUpdate(format!("margin ledger valuation: {error}"))
            })
            .and_then(|valuation| {
                let line = self
                    .settlement_asset
                    .as_ref()
                    .and_then(|asset| valuation.by_asset.get(asset).cloned())
                    .unwrap_or_default();
                self.figures_of(&line).or_overflow("run figures")
            });
        match figures {
            Ok(figures) => self.figures = figures,
            Err(error) => self.value_failed(error),
        }
        true
    }

    fn value_failed(&mut self, error: MMError) {
        self.valuation_failures = self.valuation_failures.saturating_add(1);
        self.record_fault(error);
    }

    /// One point of the equity curve, if the run can be valued.
    fn sample(&mut self, timestamp: u64) {
        if !self.value() {
            self.unvalued_ticks = self.unvalued_ticks.saturating_add(1);
            return;
        }
        // None only before any valuation succeeded, which leaves no figure to show.
        let Some(equity) = self.figures.equity else {
            return;
        };
        if self.config.record_equity_curve {
            self.equity_curve.push((timestamp, equity));
        }
        if self.stale_from_ms.is_none()
            && self
                .lanes
                .iter()
                .any(|lane| Self::rests_on_last_mark(&self.ledger, lane))
        {
            self.stale_from_ms = Some(timestamp);
        }
        if equity > self.peak_equity {
            self.peak_equity = equity;
        }
        match self.peak_equity.checked_sub(equity) {
            Some(drawdown) => self.max_drawdown = self.max_drawdown.max(drawdown),
            None => self.value_failed(MMError::NumericalError(
                "the run's drawdown overflowed".to_string(),
            )),
        }
    }

    fn rests_on_last_mark(ledger: &Ledger, lane: &Lane<D, F>) -> bool {
        lane.expired
            && ledger
                .position(&lane.ledger_id)
                .is_some_and(|p| !p.is_flat())
    }

    fn result(&self, num_ticks: u64, start_time: u64, end_time: u64) -> ContractRunResult {
        let contracts = self
            .lanes
            .iter()
            .map(|lane| ContractSummary {
                contract: lane.id().clone(),
                position: self
                    .ledger
                    .position(&lane.ledger_id)
                    .map_or(Decimal::ZERO, |p| p.quantity),
                max_position: lane.max_position,
                fills: lane.fills,
                last_mark: lane.mark.map(|(mark, _)| mark),
                last_mark_ms: lane.mark.map(|(_, at)| at),
                expired: lane.expired,
                settled: lane.settled,
                settlement_failed: lane.settlement_failed,
                expired_unsettled: Self::rests_on_last_mark(&self.ledger, lane),
                ignored_ticks: lane.ignored_ticks,
                funding: lane.funding,
                funding_missing: match &lane.terms {
                    LaneTerms::Perpetual(terms) if num_ticks > 0 => {
                        terms.missing_between(start_time, self.clock.now())
                    }
                    _ => 0,
                },
            })
            .collect();
        let never_valued = self
            .lanes
            .iter()
            .filter(|lane| {
                lane.mark.is_none()
                    && self
                        .ledger
                        .position(&lane.ledger_id)
                        .is_some_and(|p| !p.is_flat())
            })
            .map(|lane| lane.id().clone())
            .collect();
        ContractRunResult {
            total_pnl: self.figures.total_pnl,
            realized_pnl: self.figures.realized,
            total_funding: self.figures.funding,
            fundings: self.fundings.clone(),
            total_fees: self.figures.fees,
            net_pnl: self.figures.net_pnl,
            unrealized_pnl: self.figures.unrealized,
            num_trades: self.num_trades,
            num_ticks,
            start_time,
            end_time,
            equity_curve: self.equity_curve.clone(),
            trades: self.trades.clone(),
            max_drawdown: self.max_drawdown,
            sharpe_ratio: sharpe_ratio(&self.equity_curve),
            accounting: AccountingStatus {
                accountant: Some("margin-rs ledger of several contracts".to_string()),
                refused_executions: self.accounting_errors.iter().map(|(id, _)| *id).collect(),
                snapshot_failures: Vec::new(),
                valuation_failures: self.valuation_failures,
                setup_refused: self.setup_refused.clone(),
                dropped_fills: self.dropped_fills,
            },
            contracts,
            unvalued_ticks: self.unvalued_ticks,
            never_valued,
            stale_from_ms: self.stale_from_ms,
            overdrafts: self.overdrafts.clone(),
            discarded_commands: self.discarded_commands,
            fault_count: self.fault_count,
        }
    }
}
