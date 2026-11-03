//! L3 reconstruction on a deterministic synthetic stream.
//!
//! Generates 50 000 valid messages (opens, partial and full matches,
//! cancels, STP decreases, received only orders and their off book
//! `done`s) against an independent model of the book, with two injected
//! faults: a skipped sequence at message 20 000 and a match larger than its
//! maker at message 35 000. After each fault the reconstructor is
//! resynchronised from a snapshot of the model. The final visible book is
//! compared with the model's.
//!
//! Writes, into the directory given as the first argument (default
//! `target/l3_reconstruction`), `stream.csv` (every message fed, one per
//! line) and `report.txt` (the quality manifest and the comparison). Record
//! the run with the commit, the toolchain and `shasum -a 256` of both files.
//!
//! Run with `cargo run --example l3_reconstruction --features l3`.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

use market_maker_rs::Decimal;
use market_maker_rs::l3::{
    BookReconstructor, ChangeReason, DoneReason, L3Event, L3Message, L3Side, L3Snapshot,
    SnapshotOrder, VenueRules,
};

const PRODUCT: &str = "BTC-USD";
const MESSAGES: u64 = 50_000;
const GAP_AT: u64 = 20_000;
const BAD_MATCH_AT: u64 = 35_000;

/// A fixed linear congruential sequence.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }
}

type Model = BTreeMap<String, (L3Side, Decimal, Decimal)>;

fn snapshot_of(model: &Model, sequence: u64) -> L3Snapshot {
    let orders = |side: L3Side| {
        model
            .iter()
            .filter(|(_, (s, _, _))| *s == side)
            .map(|(id, (_, price, size))| SnapshotOrder {
                order_id: id.clone(),
                price: *price,
                size: *size,
            })
            .collect()
    };
    L3Snapshot {
        product_id: PRODUCT.to_string(),
        sequence,
        bids: orders(L3Side::Buy),
        asks: orders(L3Side::Sell),
    }
}

fn l2(model: &Model, side: L3Side) -> Vec<(Decimal, Decimal)> {
    let mut levels: BTreeMap<Decimal, Decimal> = BTreeMap::new();
    for (s, price, size) in model.values() {
        if *s == side {
            *levels.entry(*price).or_default() += *size;
        }
    }
    match side {
        L3Side::Buy => levels.into_iter().rev().collect(),
        L3Side::Sell => levels.into_iter().collect(),
    }
}

/// The next valid event, applied to the model.
fn next_event(rng: &mut Lcg, model: &mut Model, next_id: &mut u64) -> L3Event {
    let ids: Vec<String> = model.keys().cloned().collect();
    let roll = rng.next() % 20;
    if ids.is_empty() || roll < 8 {
        *next_id += 1;
        let id = format!("o{next_id}");
        let side = if rng.next().is_multiple_of(2) {
            L3Side::Buy
        } else {
            L3Side::Sell
        };
        let tick = Decimal::from(rng.next() % 11);
        let price = match side {
            L3Side::Buy => Decimal::from(90) + tick,
            L3Side::Sell => Decimal::from(101) + tick,
        };
        let size = Decimal::new(i64::try_from(rng.next() % 1000 + 1).unwrap_or(1), 2);
        model.insert(id.clone(), (side, price, size));
        return L3Event::Open {
            order_id: id,
            side,
            price,
            remaining: size,
        };
    }
    if roll == 8 {
        *next_id += 1;
        // Received and done without ever resting (an immediate fill); its
        // own id space, never an id on the book.
        return L3Event::Received {
            order_id: format!("r{next_id}"),
            side: L3Side::Buy,
        };
    }
    if roll == 9 {
        return L3Event::Done {
            order_id: format!("r{next_id}"),
            side: L3Side::Buy,
            price: None,
            reason: DoneReason::Filled,
            remaining: None,
        };
    }
    let id = ids
        .get(usize::try_from(rng.next()).unwrap_or(0) % ids.len())
        .cloned()
        .unwrap_or_default();
    let Some(&(side, price, size)) = model.get(&id) else {
        return L3Event::Unsupported {
            kind: "unreachable".to_string(),
        };
    };
    if roll < 15 {
        let take = Decimal::new(i64::try_from(rng.next() % 100 + 1).unwrap_or(1), 2).min(size);
        let left = size - take;
        if left == Decimal::ZERO {
            model.remove(&id);
        } else {
            model.insert(id.clone(), (side, price, left));
        }
        L3Event::Match {
            maker_id: id,
            taker_id: "taker".to_string(),
            side,
            price,
            size: take,
        }
    } else if roll < 18 {
        model.remove(&id);
        L3Event::Done {
            order_id: id,
            side,
            price: Some(price),
            reason: DoneReason::Canceled,
            remaining: Some(size),
        }
    } else {
        let new = (size * Decimal::new(5, 1)).round_dp(8);
        if new == Decimal::ZERO {
            model.remove(&id);
        } else {
            model.insert(id.clone(), (side, price, new));
        }
        L3Event::Change {
            order_id: id,
            side,
            reason: ChangeReason::Stp,
            price: Some(price),
            old_size: Some(size),
            new_size: Some(new),
            old_price: None,
            new_price: None,
            new_funds: None,
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::args()
        .nth(1)
        .map_or_else(|| PathBuf::from("target/l3_reconstruction"), PathBuf::from);
    fs::create_dir_all(&dir)?;
    let mut rng = Lcg(77);
    let mut model = Model::new();
    let mut next_id = 0_u64;
    let mut r = BookReconstructor::new(PRODUCT, VenueRules::default());
    r.bootstrap(&snapshot_of(&model, 0))?;
    let mut stream = String::from("sequence,event\n");
    let mut seq = 0_u64;
    let mut errors = String::new();
    for i in 1..=MESSAGES {
        seq += 1;
        if i == GAP_AT {
            // Skip a sequence: the stream jumps by one.
            seq += 1;
        }
        let event = if i == BAD_MATCH_AT {
            // A match larger than its maker; the model is left as it was.
            let (id, (side, price, size)) = model
                .iter()
                .next()
                .map(|(k, v)| (k.clone(), *v))
                .ok_or("empty model")?;
            L3Event::Match {
                maker_id: id,
                taker_id: "taker".to_string(),
                side,
                price,
                size: size + Decimal::ONE,
            }
        } else {
            next_event(&mut rng, &mut model, &mut next_id)
        };
        let message = L3Message {
            product_id: PRODUCT.to_string(),
            sequence: seq,
            time_us: 1_700_000_000_000_000,
            event,
        };
        writeln!(stream, "{seq},{:?}", message.event)?;
        if let Err(error) = r.apply(&message) {
            writeln!(errors, "{error:?}")?;
            // Resynchronise from the model, which already reflects this
            // message unless it was the injected bad match.
            r.bootstrap(&snapshot_of(&model, seq))?;
        }
        r.audit()?;
    }
    let bids_match = r.l2(L3Side::Buy)? == l2(&model, L3Side::Buy);
    let asks_match = r.l2(L3Side::Sell)? == l2(&model, L3Side::Sell);
    let manifest = r.manifest();
    let mut report = String::new();
    writeln!(report, "messages: {MESSAGES}")?;
    writeln!(
        report,
        "injected: gap at message {GAP_AT}, oversized match at message {BAD_MATCH_AT}"
    )?;
    writeln!(report, "errors:\n{errors}")?;
    writeln!(report, "counts: {:?}", manifest.counts)?;
    writeln!(report, "segments:")?;
    for s in &manifest.segments {
        writeln!(report, "  {s:?}")?;
    }
    writeln!(
        report,
        "final L2 equals the model: bids {bids_match}, asks {asks_match}"
    )?;
    writeln!(
        report,
        "final levels: {} bids, {} asks",
        r.l2(L3Side::Buy)?.len(),
        r.l2(L3Side::Sell)?.len()
    )?;
    fs::write(dir.join("stream.csv"), &stream)?;
    fs::write(dir.join("report.txt"), &report)?;
    print!("{report}");
    if !(bids_match && asks_match) {
        return Err("the reconstructed book differs from the model".into());
    }
    Ok(())
}
