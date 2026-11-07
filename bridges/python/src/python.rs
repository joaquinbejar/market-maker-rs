//! The extension module: a thin layer of Python values over [`Worker`].

use std::sync::{Mutex, PoisonError};

use market_maker_rs::Decimal;
use market_maker_rs::backtest::kernel::{
    ActionMask, ActionStatus, KernelAction, KernelTick, QuoteSide,
};
use market_maker_rs::rl::{FeatureVector, ResidualProposal, RlAction, SpecId};
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::worker::{BridgeConfig, BridgeError, ResetReply, StepReply, Worker, refused_because};

create_exception!(_native, BridgeFailure, PyException);

fn to_py(error: BridgeError) -> PyErr {
    match error {
        BridgeError::Refused(reason) => PyValueError::new_err(reason),
        other => BridgeFailure::new_err(other.to_string()),
    }
}

fn hex(value: u128) -> String {
    format!("{value:032x}")
}

fn mask_dict<'py>(py: Python<'py>, mask: &ActionMask) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("quote_ok", mask.global_quote_ok)?;
    out.set_item("bid", mask.bid)?;
    out.set_item("ask", mask.ask)?;
    out.set_item("size", mask.size)?;
    out.set_item("refusal", mask.refusal.map(|reason| format!("{reason:?}")))?;
    Ok(out)
}

fn observation_dict<'py>(py: Python<'py>, obs: &FeatureVector) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("row", obs.row)?;
    out.set_item("ts_ns", obs.ts_ns)?;
    out.set_item("values", obs.values.to_vec())?;
    let raw: Vec<Option<String>> = obs
        .raw
        .iter()
        .map(|value| value.map(|exact| exact.to_string()))
        .collect();
    out.set_item("raw", raw)?;
    Ok(out)
}

fn exact(value: Decimal) -> String {
    value.normalize().to_string()
}

fn reset_dict<'py>(py: Python<'py>, reply: &ResetReply) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("observation", observation_dict(py, &reply.observation)?)?;
    out.set_item("mask", mask_dict(py, &reply.mask)?)?;
    out.set_item("hedge_mask", reply.hedge_mask)?;
    out.set_item("done", reply.done)?;
    Ok(out)
}

fn action_dict<'py>(py: Python<'py>, action: &KernelAction) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    match action {
        KernelAction::Hold => out.set_item("kind", "hold")?,
        KernelAction::Quote {
            bid,
            ask,
            bid_lots,
            ask_lots,
        } => {
            out.set_item("kind", "quote")?;
            out.set_item("bid", bid)?;
            out.set_item("ask", ask)?;
            out.set_item("bid_lots", bid_lots)?;
            out.set_item("ask_lots", ask_lots)?;
        }
        KernelAction::QuoteError(code) => {
            out.set_item("kind", "strategy_error")?;
            out.set_item("code", code.0)?;
        }
        KernelAction::OneSided { side, price, lots } => {
            out.set_item(
                "kind",
                match side {
                    QuoteSide::Bid => "bid",
                    QuoteSide::Ask => "ask",
                },
            )?;
            out.set_item("price", price)?;
            out.set_item("lots", lots)?;
        }
        KernelAction::Cancel => out.set_item("kind", "cancel")?,
    }
    Ok(out)
}

fn step_dict<'py>(py: Python<'py>, reply: &StepReply) -> PyResult<Bound<'py, PyDict>> {
    let t = &reply.transition;
    let out = PyDict::new(py);
    out.set_item("observation", observation_dict(py, &t.observation)?)?;
    out.set_item("mask", mask_dict(py, &t.mask)?)?;
    out.set_item("hedge_mask", t.hedge_mask)?;
    out.set_item("reward_f32", t.reward.as_f32)?;
    out.set_item("reward", exact(t.reward.total))?;
    out.set_item("ledger_change", exact(t.reward.ledger_change))?;
    out.set_item("terminal_close", exact(t.reward.terminal_close))?;
    out.set_item("penalty", exact(t.reward.penalty))?;
    out.set_item("net", exact(reply.net))?;
    out.set_item("penalties", exact(reply.penalties))?;
    let (applied, reason) = match t.action_status {
        ActionStatus::Applied => (true, None),
        ActionStatus::Rejected(reason) => (false, Some(format!("{reason:?}"))),
        ActionStatus::QuoteError(_) => (false, refused_because(t.action_status)),
    };
    out.set_item("applied", applied)?;
    out.set_item("refused_because", reason)?;
    out.set_item("terminated", t.terminated)?;
    out.set_item("truncated", t.truncated)?;
    match &t.close {
        Some(close) => {
            let closed = PyDict::new(py);
            closed.set_item("side", format!("{:?}", close.side))?;
            closed.set_item("price", exact(close.price))?;
            closed.set_item("quantity", exact(close.quantity))?;
            closed.set_item("fee", exact(close.fee))?;
            closed.set_item("beyond_touch", exact(close.beyond_touch))?;
            out.set_item("close", closed)?;
        }
        None => out.set_item("close", py.None())?,
    }
    if let Some(decision) = &reply.decision {
        let log = PyDict::new(py);
        log.set_item("classical", action_dict(py, &decision.classical)?)?;
        log.set_item("composed", action_dict(py, &decision.composed)?)?;
        let d = decision.delta;
        log.set_item(
            "delta",
            (d.bid_away_ticks, d.ask_away_ticks, d.bid_lots, d.ask_lots),
        )?;
        let c = decision.clips;
        log.set_item("clips", (c.proposal, c.price, c.size))?;
        log.set_item(
            "fallback",
            decision.fallback.map(|reason| format!("{reason:?}")),
        )?;
        out.set_item("residual", log)?;
    }
    Ok(out)
}

/// One environment on its own thread. The Python lock is released while it works.
#[pyclass]
struct NativeEnv {
    worker: Mutex<Worker>,
}

impl NativeEnv {
    fn with_worker<T: Send>(
        &self,
        py: Python<'_>,
        call: impl FnOnce(&mut Worker) -> Result<T, BridgeError> + Send,
    ) -> PyResult<T> {
        py.detach(|| {
            let mut worker = self.worker.lock().unwrap_or_else(PoisonError::into_inner);
            call(&mut worker)
        })
        .map_err(to_py)
    }

    fn take_step(&self, py: Python<'_>, action: RlAction) -> PyResult<Py<PyDict>> {
        let reply = self.with_worker(py, |worker| worker.step(action))?;
        Ok(step_dict(py, &reply)?.unbind())
    }
}

#[pymethods]
impl NativeEnv {
    /// Starts an environment. `rows` hold time in nanoseconds, then bid and bid size, then
    /// ask and ask size, in ticks and lots. `expected` is `(observation, action, reward,
    /// episode, fingerprint in hex)`.
    #[new]
    #[pyo3(signature = (config_json, rows, expected=None))]
    fn new(
        py: Python<'_>,
        config_json: &str,
        rows: Vec<(u64, u64, u64, u64, u64)>,
        expected: Option<(u16, u16, u16, u16, String)>,
    ) -> PyResult<Self> {
        let config = BridgeConfig::from_json(config_json).map_err(to_py)?;
        let expected = expected
            .map(|(observation, action, reward, episode, fingerprint)| {
                u128::from_str_radix(&fingerprint, 16)
                    .map(|fingerprint| SpecId {
                        observation,
                        action,
                        reward,
                        episode,
                        fingerprint,
                    })
                    .map_err(|error| PyValueError::new_err(format!("fingerprint: {error}")))
            })
            .transpose()?;
        let rows: Vec<KernelTick> = rows
            .into_iter()
            .map(|(ts_ns, bid, bid_lots, ask, ask_lots)| KernelTick {
                ts_ns,
                bid,
                bid_lots,
                ask,
                ask_lots,
            })
            .collect();
        let worker = py
            .detach(move || Worker::spawn(config, rows, expected))
            .map_err(to_py)?;
        Ok(Self {
            worker: Mutex::new(worker),
        })
    }

    /// What this environment is.
    fn info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let info = self.with_worker(py, |worker| Ok(worker.info()))?;
        let out = PyDict::new(py);
        out.set_item(
            "spec",
            (
                info.spec.observation,
                info.spec.action,
                info.spec.reward,
                info.spec.episode,
                hex(info.spec.fingerprint),
            ),
        )?;
        out.set_item("features", info.features)?;
        out.set_item("offsets", info.offsets)?;
        out.set_item("sizes", info.sizes)?;
        out.set_item("hedge_actions", info.hedge_actions)?;
        out.set_item("cancel_action", info.cancel_action)?;
        out.set_item("kernel_fingerprint", hex(info.kernel_fingerprint))?;
        out.set_item("dataset_digest", hex(info.dataset_digest))?;
        out.set_item("rows", info.rows)?;
        Ok(out)
    }

    /// Starts an episode.
    fn reset<'py>(&self, py: Python<'py>, seed: u64) -> PyResult<Bound<'py, PyDict>> {
        let reply = self.with_worker(py, |worker| worker.reset(seed))?;
        reset_dict(py, &reply)
    }

    /// Leaves the resting orders in place.
    fn step_hold(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        self.take_step(py, RlAction::Hold)
    }

    /// Replaces the resting orders with a bid and an ask, by grid index.
    fn step_quote(
        &self,
        py: Python<'_>,
        bid: usize,
        ask: usize,
        bid_size: usize,
        ask_size: usize,
    ) -> PyResult<Py<PyDict>> {
        self.take_step(
            py,
            RlAction::Quote {
                bid,
                ask,
                bid_size,
                ask_size,
            },
        )
    }

    /// Replaces the resting orders with one order, on the bid if `bid` is true.
    fn step_one_sided(
        &self,
        py: Python<'_>,
        bid: bool,
        offset: usize,
        size: usize,
    ) -> PyResult<Py<PyDict>> {
        let side = if bid { QuoteSide::Bid } else { QuoteSide::Ask };
        self.take_step(py, RlAction::OneSided { side, offset, size })
    }

    /// Cancels every resting or pending order and sends nothing.
    fn step_cancel(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        self.take_step(py, RlAction::Cancel)
    }

    /// Reduces the delivered position with a marketable order at the observed touch, of the
    /// size at `size` in the grid's sizes.
    fn step_hedge(&self, py: Python<'_>, size: usize) -> PyResult<Py<PyDict>> {
        self.take_step(py, RlAction::Hedge { size })
    }

    /// Takes one decision of a residual policy.
    fn step_residual(
        &self,
        py: Python<'_>,
        bid_away: f64,
        ask_away: f64,
        bid_size: f64,
        ask_size: f64,
    ) -> PyResult<Py<PyDict>> {
        let proposal = ResidualProposal {
            bid_away,
            ask_away,
            bid_size,
            ask_size,
        };
        let reply = self.with_worker(py, |worker| worker.step_residual(proposal))?;
        Ok(step_dict(py, &reply)?.unbind())
    }

    /// The attribution of the episode so far, reconciled with the reward.
    fn attribution<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let reply = self.with_worker(py, Worker::attribution)?;
        let a = reply.attribution;
        let out = PyDict::new(py);
        out.set_item("spread_capture", exact(a.spread_capture))?;
        out.set_item("inventory_pnl", exact(a.inventory_pnl))?;
        out.set_item("quote_fees", exact(a.quote_fees))?;
        out.set_item("base_fees", exact(a.base_fees))?;
        out.set_item("net_pnl", exact(a.net_pnl))?;
        out.set_item("fills", reply.fills)?;
        let inventory = reply.inventory;
        out.set_item("max_abs_position", exact(inventory.max_abs))?;
        out.set_item("mean_abs_position", inventory.mean_abs.map(exact))?;
        out.set_item("final_position", exact(inventory.final_position))?;
        out.set_item("max_drawdown", exact(reply.max_drawdown))?;
        out.set_item("window_ms", vec![reply.window_ms.0, reply.window_ms.1])?;
        let positions: Vec<(u64, String, String)> = reply
            .positions
            .iter()
            .map(|p| (p.timestamp_ms, exact(p.price), exact(p.position)))
            .collect();
        out.set_item("positions", positions)?;
        Ok(out)
    }

    /// Ends the thread and frees the dataset.
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        self.with_worker(py, Worker::close)
    }

    /// Returns true once the worker has ended.
    fn is_closed(&self, py: Python<'_>) -> PyResult<bool> {
        self.with_worker(py, |worker| Ok(worker.is_closed()))
    }
}

#[pymodule]
fn _native(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeEnv>()?;
    module.add("BridgeFailure", py.get_type::<BridgeFailure>())?;
    module.add("RESIDUAL_VERSION", market_maker_rs::rl::RESIDUAL_VERSION)?;
    module.add("BRIDGE_VERSION", env!("CARGO_PKG_VERSION"))?;
    module.add("LIBRARY_VERSION", market_maker_rs::rl::LIBRARY_VERSION)?;
    Ok(())
}
