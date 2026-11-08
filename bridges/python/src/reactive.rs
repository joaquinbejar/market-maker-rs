//! A queue-reactive day whose market reacts to the agent's fills (#282), for experiments.
//!
//! [`ReactiveRows`] builds the rows `qr_days` writes for one regime and day and keeps the
//! simulator's state at each row, through `market_maker_rs::rl::ReactiveQrMarket`. After a
//! row it imposes reference moves, as a trade's permanent impact, and simulates every later
//! row again; the rows up to it do not change. How many moves a fill causes is the
//! experiment's: this only carries them out. It builds and tests without Python.

use market_maker_rs::Decimal;
use market_maker_rs::backtest::kernel::{KernelConfig, KernelDataset};
use market_maker_rs::backtest::queue_reactive::{QrParameters, UnsupportedPolicy};
use market_maker_rs::rl::{QrEpisode, ReactiveQrMarket};
use serde::Deserialize;

/// A regime as #102's design file states it.
#[derive(Debug, Deserialize)]
struct Regime {
    tick_size: String,
    theta: f64,
    theta_reinit: f64,
    event_sizes: Vec<String>,
    rates: Vec<Vec<[f64; 3]>>,
    policy: String,
}

/// The kernel's grid, as #102's design file states it.
#[derive(Debug, Deserialize)]
struct Kernel {
    tick_size: String,
    lot_size: String,
    max_price: String,
    max_order_size: String,
}

/// One day.
#[derive(Debug, Deserialize)]
struct Episode {
    seed: u64,
    reference_ticks: i64,
    start_ms: u64,
    sample_ms: u32,
    rows: u32,
}

fn decimal(what: &str, text: &str) -> Result<Decimal, String> {
    text.parse::<Decimal>()
        .map_err(|error| format!("{what} {text:?}: {error}"))
}

fn json<'a, T: Deserialize<'a>>(what: &str, text: &'a str) -> Result<T, String> {
    serde_json::from_str(text).map_err(|error| format!("{what}: {error}"))
}

/// One regime's day of rows that react to the agent.
#[derive(Debug, Clone)]
pub struct ReactiveRows {
    market: ReactiveQrMarket,
    config: KernelConfig,
}

impl ReactiveRows {
    /// The day `episode` of `regime` on the kernel's grid `kernel`, each as JSON in #102's
    /// design's form; `episode` holds `seed`, `reference_ticks`, `start_ms`, `sample_ms`
    /// and `rows`.
    ///
    /// # Errors
    ///
    /// A message for JSON that does not parse, parameters or a grid the simulator or the
    /// kernel refuses, or a day that does not simulate.
    pub fn new(regime: &str, kernel: &str, episode: &str) -> Result<Self, String> {
        let regime: Regime = json("regime", regime)?;
        let kernel: Kernel = json("kernel", kernel)?;
        let episode: Episode = json("episode", episode)?;
        let sizes = regime
            .event_sizes
            .iter()
            .map(|size| decimal("event size", size))
            .collect::<Result<Vec<_>, _>>()?;
        let rates: Vec<Vec<(f64, f64, f64)>> = regime
            .rates
            .iter()
            .map(|table| table.iter().map(|[l, c, m]| (*l, *c, *m)).collect())
            .collect();
        let supported = rates.iter().map(|table| vec![true; table.len()]).collect();
        let policy = match regime.policy.as_str() {
            "stop" => UnsupportedPolicy::Stop,
            "nearest_supported" => UnsupportedPolicy::NearestSupported,
            other => return Err(format!("unknown policy {other}")),
        };
        let params = QrParameters::new(
            decimal("tick_size", &regime.tick_size)?,
            regime.theta,
            regime.theta_reinit,
            sizes,
            rates,
            supported,
            policy,
        )
        .map_err(|error| error.to_string())?;
        let config = KernelConfig::new(
            decimal("tick_size", &kernel.tick_size)?,
            decimal("lot_size", &kernel.lot_size)?,
            decimal("max_price", &kernel.max_price)?,
            decimal("max_order_size", &kernel.max_order_size)?,
            u64::from(episode.rows),
        );
        let day = QrEpisode {
            seed: episode.seed,
            reference_ticks: episode.reference_ticks,
            start_ms: episode.start_ms,
            sample_ms: episode.sample_ms,
            rows: episode.rows,
        };
        let market =
            ReactiveQrMarket::new(&params, &day, &config).map_err(|error| error.to_string())?;
        Ok(Self { market, config })
    }

    /// The rows as the bridge reads them: `[ts_ns, bid, bid_lots, ask, ask_lots]`, in the
    /// kernel's ticks and lots.
    ///
    /// # Errors
    ///
    /// A message for a row the kernel refuses.
    pub fn rows(&self) -> Result<Vec<[u64; 5]>, String> {
        let market = self.market.market().map_err(|error| error.to_string())?;
        Ok(market
            .dataset
            .ticks()
            .iter()
            .map(|t| [t.ts_ns, t.bid, t.bid_lots, t.ask, t.ask_lots])
            .collect())
    }

    /// The digest the bridge reports for these rows, as `qr_days` prints it.
    ///
    /// # Errors
    ///
    /// As [`ReactiveRows::rows`].
    pub fn digest(&self) -> Result<u128, String> {
        let market = self.market.market().map_err(|error| error.to_string())?;
        let seen = KernelDataset::from_kernel_ticks(market.dataset.ticks().to_vec(), &self.config)
            .map_err(|error| error.to_string())?;
        Ok(seen.id().digest)
    }

    /// Imposes `moves` reference moves after row `row`, up for a positive count and down for
    /// a negative one, and simulates every later row again. No moves change nothing. Returns
    /// the first-limit volume the moves emptied, as exact text.
    ///
    /// # Errors
    ///
    /// A message for a row past the last, more moves than
    /// `market_maker_rs::rl::MAX_REACTION_MOVES`, or a move the simulator cannot make.
    pub fn react(&mut self, row: u64, moves: i64) -> Result<String, String> {
        let row = usize::try_from(row).map_err(|_| format!("row {row} is out of range"))?;
        self.market
            .react(row, moves)
            .map(|emptied| emptied.normalize().to_string())
            .map_err(|error| error.to_string())
    }

    /// Why the simulator stopped before the last row, if it did.
    #[must_use]
    pub fn stopped(&self) -> Option<String> {
        self.market.stopped().map(|step| format!("{step:?}"))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use serde_json::Value;

    const DAY_MS: u64 = 86_400_000;

    fn design() -> Value {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/experiments/m4_05/design.json");
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
    }

    fn day(regime: usize, day: u64, rows: u32) -> ReactiveRows {
        let design = design();
        let simulator = &design["simulator"];
        let episode = serde_json::json!({
            "seed": 1_000_003 * (regime as u64 + 1) + day,
            "reference_ticks": simulator["reference_ticks"],
            "start_ms": (day - 1) * DAY_MS,
            "sample_ms": simulator["sample_ms"],
            "rows": rows,
        });
        ReactiveRows::new(
            &simulator["regimes"][regime].to_string(),
            &design["kernel"].to_string(),
            &episode.to_string(),
        )
        .unwrap()
    }

    #[test]
    fn test_rows_and_digest_are_qr_days_for_a_design_day() {
        let digests: Value = serde_json::from_slice(
            &std::fs::read(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/experiments/m4_05/digests.json"
            ))
            .unwrap(),
        )
        .unwrap();
        let record = &digests["days"][0];
        assert_eq!(
            (record["regime"].as_str(), record["day"].as_u64()),
            (Some("calm"), Some(1))
        );
        let rows = day(0, 1, 8640);
        assert_eq!(
            format!("{:032x}", rows.digest().unwrap()),
            record["dataset_digest"].as_str().unwrap()
        );
        assert_eq!(rows.rows().unwrap().len(), 8640);
        assert!(rows.stopped().is_none());
    }

    #[test]
    fn test_a_reaction_keeps_the_rows_up_to_it_and_none_changes_nothing() {
        let original = day(1, 3, 600);
        let mut same = original.clone();
        same.react(100, 0).unwrap();
        assert_eq!(same.rows().unwrap(), original.rows().unwrap());
        let mut moved = original.clone();
        moved.react(100, 3).unwrap();
        let (a, b) = (original.rows().unwrap(), moved.rows().unwrap());
        assert_eq!(a.len(), b.len());
        assert_eq!(a[..=100], b[..=100]);
        assert_ne!(a[101..], b[101..]);
        let mut again = original.clone();
        again.react(100, 3).unwrap();
        assert_eq!(
            again.rows().unwrap(),
            b,
            "the same moves reproduce the same rows"
        );
    }

    #[test]
    fn test_bad_input_is_a_message_never_a_panic() {
        let design = design();
        let regime = design["simulator"]["regimes"][0].to_string();
        let kernel = design["kernel"].to_string();
        let episode = r#"{"seed": 1, "reference_ticks": 10000, "start_ms": 0, "sample_ms": 10000, "rows": 10}"#;
        assert!(ReactiveRows::new("{", &kernel, episode).is_err());
        assert!(ReactiveRows::new(&regime, "{}", episode).is_err());
        assert!(ReactiveRows::new(&regime, &kernel, r#"{"seed": -1}"#).is_err());
        let zero =
            r#"{"seed": 1, "reference_ticks": 10000, "start_ms": 0, "sample_ms": 0, "rows": 10}"#;
        assert!(ReactiveRows::new(&regime, &kernel, zero).is_err());
        let mut rows = ReactiveRows::new(&regime, &kernel, episode).unwrap();
        assert!(rows.react(10, 1).is_err(), "past the last row");
        assert!(rows.react(u64::MAX, 1).is_err());
        assert!(rows.react(9, i64::MIN).is_err(), "too many moves");
        assert!(rows.react(9, i64::MAX).is_err());
    }
}
