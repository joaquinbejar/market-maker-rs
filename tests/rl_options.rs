#![cfg(feature = "rl")]
//! The options quoting spaces and the net delta: behaviour, and the published fixtures
//! in `doc/fixtures/rl_options.json`. All data here is synthetic.
//!
//! Run with `MM_WRITE_RL_OPTIONS_FIXTURES=1` to rewrite the fixture file after a deliberate
//! change of a version constant. The same file is checked on `x86_64` and `aarch64`, so a
//! delta that differs by one bit between them fails here.

use std::str::FromStr;

use market_maker_rs::Decimal;
use market_maker_rs::backtest::kernel::{Observation, Status};
use market_maker_rs::dec;
use market_maker_rs::prelude::OptionStyle;
use market_maker_rs::rl::{
    AgentInput, Lane, LinearPosition, ListedOption, NET_DELTA_VERSION,
    OPTIONS_ACTION_SPACE_VERSION, OPTIONS_OBSERVATION_SPACE_VERSION, ObservationScales,
    OptionPosition, OptionsAction, OptionsKind, OptionsProfile, OptionsScales, PreviousRow,
    RL_OPTIONS_FIXTURES, SlotInput, black_scholes_delta, black_scholes_delta_unrounded,
    hedge_cap_lots,
};
use market_maker_rs::types::error::MMError;
use serde_json::{Value, json};

const FIXTURE: &str = "doc/fixtures/rl_options.json";
const DAY_NS: u64 = 86_400_000_000_000;

fn hedge_scales() -> ObservationScales {
    ObservationScales {
        spread_ticks: dec!(10),
        mid_change_ticks: dec!(10),
        row_gap_ms: dec!(1000),
        size_lots: dec!(10),
        position: dec!(10),
        entry_gap: dec!(100),
    }
}

fn scales() -> OptionsScales {
    OptionsScales::new(
        dec!(1000),
        dec!(86400000),
        dec!(20),
        dec!(50),
        dec!(500),
        dec!(20),
        dec!(10000),
        dec!(10),
    )
}

/// The proposed window: two expiry ranks and five strike ranks.
fn proposed() -> OptionsProfile {
    OptionsProfile::new(
        2,
        5,
        dec!(0.6),
        scales(),
        hedge_scales(),
        vec![1, 2, 4, 8],
        vec![1, 2, 4, 8],
        vec![1, 2, 5],
        2,
        17,
        1_000_000_000,
    )
}

fn small() -> OptionsProfile {
    OptionsProfile::new(
        1,
        2,
        dec!(0.8),
        scales(),
        hedge_scales(),
        vec![1, 3],
        vec![2],
        vec![1],
        1,
        100,
        500_000_000,
    )
}

fn style_name(style: OptionStyle) -> &'static str {
    match style {
        OptionStyle::Call => "call",
        OptionStyle::Put => "put",
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn opt(v: Option<Decimal>) -> Value {
    v.map_or(Value::Null, |d| json!(d.normalize().to_string()))
}

fn profile_json(name: &str, p: &OptionsProfile) -> Value {
    let id = p.id().unwrap();
    json!({
        "name": name,
        "expiry_ranks": p.expiry_ranks,
        "strike_ranks": p.strike_ranks,
        "sigma_delta": p.sigma_delta.to_string(),
        "bid_offsets_ticks": p.bid_offsets_ticks,
        "ask_offsets_ticks": p.ask_offsets_ticks,
        "sizes_lots": p.sizes_lots,
        "max_orders": p.max_orders,
        "max_steps": p.max_steps,
        "step_ns": p.step_ns,
        "spec_bytes": hex(&p.to_bytes().unwrap()),
        "spec_id": {
            "observation": id.observation,
            "action": id.action,
            "reward": id.reward,
            "episode": id.episode,
            "fingerprint": id.fingerprint.to_string(),
        },
        "observation_len": p.observation_len(),
        "action_head_sizes": p.action_head_sizes(),
    })
}

/// Three expiries, two classes at the same instant, a missing put and two strikes at the
/// same distance from the mark.
fn listing() -> Vec<ListedOption> {
    let mut out = Vec::new();
    for (symbol, expiry) in [("W1", 3 * DAY_NS), ("D1", DAY_NS), ("D1B", DAY_NS)] {
        for strike in [
            dec!(90),
            dec!(95),
            dec!(100),
            dec!(105),
            dec!(110),
            dec!(120),
        ] {
            for style in [OptionStyle::Call, OptionStyle::Put] {
                if symbol == "D1" && strike == dec!(105) && style == OptionStyle::Put {
                    continue;
                }
                out.push(ListedOption::new(symbol, expiry, strike, style));
            }
        }
    }
    out
}

fn slot_cases() -> Value {
    let p = proposed();
    let listed = listing();
    let mut cases = Vec::new();
    for (name, mark) in [
        ("no_mark", None),
        ("mark_100", Some(dec!(100))),
        ("mark_102.5_ties_to_lower", Some(dec!(102.5))),
    ] {
        let slots = p.resolve_slots(mark, &listed).unwrap();
        let named: Vec<Value> = slots
            .iter()
            .map(|s| {
                s.map_or(Value::Null, |i| {
                    let l = &listed[i];
                    json!([l.expiry_symbol, l.strike.to_string(), style_name(l.style)])
                })
            })
            .collect();
        cases.push(json!({ "name": name, "index_mark": opt(mark), "slots": named }));
    }
    let one_expiry: Vec<ListedOption> = listed
        .iter()
        .filter(|l| l.expiry_symbol == "W1")
        .cloned()
        .collect();
    let slots = p.resolve_slots(Some(dec!(100)), &one_expiry).unwrap();
    let named: Vec<Value> = slots
        .iter()
        .map(|s| {
            s.map_or(Value::Null, |i| {
                let l = &one_expiry[i];
                json!([l.expiry_symbol, l.strike.to_string(), style_name(l.style)])
            })
        })
        .collect();
    cases.push(json!({
        "name": "only_W1_listed",
        "index_mark": "100",
        "listed_filter": "W1",
        "slots": named,
    }));
    let listed_json: Vec<Value> = listed
        .iter()
        .map(|l| {
            json!([
                l.expiry_symbol,
                l.expiry_ns,
                l.strike.to_string(),
                style_name(l.style)
            ])
        })
        .collect();
    json!({ "profile": "proposed", "listed": listed_json, "cases": cases })
}

fn hedge_obs() -> (Observation, Option<PreviousRow>, Decimal) {
    let obs = Observation {
        row: Some(7),
        ts_ns: 5_000_000_000,
        bid: 200_000,
        bid_lots: 3,
        ask: 200_002,
        ask_lots: 4,
        position: dec!(-0.3),
        avg_entry: dec!(100001),
        delivered_fills: 0,
        delivered_local_rejects: 0,
        open_orders: 1,
        awaiting_cancel_outcome: 0,
        step: 4,
        status: Status::Running,
        rejected: [0; 8],
    };
    (
        obs,
        Some(PreviousRow::new(399_998, 4_000_000_000)),
        dec!(0.5),
    )
}

fn observation_case() -> Value {
    let p = small();
    let decision = 2 * DAY_NS / 3;
    let slots = vec![
        Some(
            SlotInput::new(dec!(100), DAY_NS, dec!(0.1))
                .with_book(Some((dec!(4.2), dec!(10))), Some((dec!(4.6), dec!(7))))
                .with_venue_mark(dec!(4.41))
                .with_account(dec!(-3), [1, 0], [0, 1]),
        ),
        Some(
            SlotInput::new(dec!(100), DAY_NS, dec!(0.1))
                .with_book(None, Some((dec!(5.5), dec!(80)))),
        ),
        None,
        Some(
            SlotInput::new(dec!(95), DAY_NS, dec!(0.1))
                .with_book(Some((dec!(1.0), dec!(2))), Some((dec!(1.3), dec!(2))))
                .with_account(dec!(2), [0, 0], [0, 0]),
        ),
    ];
    let agent = AgentInput::new(dec!(-12.5), Some(dec!(-0.731)), 4);
    let (obs, prev, tick) = hedge_obs();
    let out = p
        .observation(
            decision,
            Some(dec!(99.8)),
            &slots,
            (&obs, prev, tick),
            &agent,
        )
        .unwrap();
    assert_eq!(out.values.len(), p.observation_len());
    json!({
        "profile": "small",
        "decision_ns": decision,
        "index_mark": "99.8",
        "raw": out.raw.iter().map(|v| opt(*v)).collect::<Vec<_>>(),
        "values": out.values.iter().map(|v| format!("{v}")).collect::<Vec<_>>(),
    })
}

fn action_cases() -> Value {
    let p = small();
    let rows: [[usize; 5]; 10] = [
        [0, 0, 1, 0, 0],
        [1, 3, 1, 0, 0],
        [2, 1, 1, 0, 0],
        [3, 2, 0, 0, 0],
        [4, 4, 0, 0, 0],
        [5, 4, 0, 0, 0],
        [4, 0, 0, 0, 0],
        [1, 4, 0, 0, 0],
        [1, 0, 2, 0, 0],
        [6, 0, 0, 0, 0],
    ];
    let cases: Vec<Value> = rows
        .iter()
        .map(|row| match p.decode_action(row) {
            Ok(a) => json!({
                "row": row,
                "kind": a.kind.code(),
                "lane": match a.lane { Lane::Option(s) => json!(s), Lane::Hedge => json!("hedge"), _ => unreachable!() },
                "canonical": p.encode_action(&a).unwrap(),
            }),
            Err(MMError::InvalidAction(_)) => json!({ "row": row, "invalid_action": true }),
            Err(e) => panic!("unexpected error {e:?}"),
        })
        .collect();
    json!({ "profile": "small", "cases": cases })
}

fn delta_cases() -> Value {
    let mut cases = Vec::new();
    let spots = [dec!(50), dec!(99.8), dec!(100), dec!(100.25), dec!(180)];
    let strikes = [dec!(80), dec!(100), dec!(125)];
    let taus = [
        dec!(-0.001),
        dec!(0),
        dec!(0.0000000317),
        dec!(0.0027397260273972602739726027),
        dec!(0.25),
        dec!(2),
    ];
    let sigmas = [dec!(0.05), dec!(0.6), dec!(1.5)];
    for s in spots {
        for k in strikes {
            for t in taus {
                for v in sigmas {
                    for style in [OptionStyle::Call, OptionStyle::Put] {
                        let d = black_scholes_delta(s, k, t, v, style).unwrap();
                        let bits = black_scholes_delta_unrounded(s, k, t, v, style)
                            .unwrap()
                            .map_or((Value::Null, Value::Null), |(d1, raw)| {
                                (
                                    json!(format!("{:016x}", d1.to_bits())),
                                    json!(format!("{:016x}", raw.to_bits())),
                                )
                            });
                        cases.push(json!([
                            s.to_string(),
                            k.to_string(),
                            t.to_string(),
                            v.to_string(),
                            style_name(style),
                            d.to_string(),
                            bits.0,
                            bits.1
                        ]));
                    }
                }
            }
        }
    }
    json!(cases)
}

fn net_delta_case(index_mark: Decimal, extra_lots: Decimal) -> Value {
    let p = proposed();
    let decision = DAY_NS / 2;
    let linear = [
        LinearPosition::new(dec!(-3), dec!(0.1)),
        LinearPosition::new(dec!(1), dec!(0.01)),
    ];
    let options = [
        OptionPosition::new(dec!(5), dec!(0.1), dec!(100), OptionStyle::Call, DAY_NS),
        OptionPosition::new(dec!(-2), dec!(0.1), dec!(95), OptionStyle::Put, 3 * DAY_NS),
        OptionPosition::new(dec!(4), dec!(0.1), dec!(105), OptionStyle::Call, DAY_NS / 4),
        OptionPosition::new(dec!(0), dec!(0.1), dec!(110), OptionStyle::Call, DAY_NS),
        OptionPosition::new(
            extra_lots,
            dec!(0.1),
            dec!(100),
            OptionStyle::Call,
            2 * DAY_NS,
        ),
    ];
    let nd = p
        .net_delta(index_mark, decision, DAY_NS, &linear, &options)
        .unwrap();
    let cap = hedge_cap_lots(nd, dec!(0.1)).unwrap();
    json!({
        "profile": "proposed",
        "index_mark": index_mark.to_string(),
        "decision_ns": decision,
        "day_ns": DAY_NS,
        "linear": linear.iter().map(|l| json!([l.lots.to_string(), l.lot_size.to_string()])).collect::<Vec<_>>(),
        "options": options.iter().map(|o| json!([o.lots.to_string(), o.base_per_lot.to_string(), o.strike.to_string(), style_name(o.style), o.expiry_ns])).collect::<Vec<_>>(),
        "net_delta": nd.to_string(),
        "hedge_lot_size": "0.1",
        "hedge_cap_lots": cap,
    })
}

fn fixtures() -> Value {
    json!({
        "format": 1,
        "observation_version": OPTIONS_OBSERVATION_SPACE_VERSION,
        "action_version": OPTIONS_ACTION_SPACE_VERSION,
        "net_delta_version": NET_DELTA_VERSION,
        "profiles": [profile_json("proposed", &proposed()), profile_json("small", &small())],
        "slots": slot_cases(),
        "observation": observation_case(),
        "actions": action_cases(),
        "deltas": delta_cases(),
        "net_delta": [
            net_delta_case(dec!(99.8), dec!(0)),
            net_delta_case(dec!(101), dec!(-40)),
        ],
    })
}

#[test]
fn fixtures_match_the_published_file() {
    let built = fixtures();
    let text = format!("{}\n", serde_json::to_string_pretty(&built).unwrap());
    if std::env::var_os("MM_WRITE_RL_OPTIONS_FIXTURES").is_some() {
        std::fs::write(FIXTURE, text).unwrap();
        return;
    }
    let on_disk: Value = serde_json::from_str(&std::fs::read_to_string(FIXTURE).unwrap()).unwrap();
    assert_eq!(
        built, on_disk,
        "{FIXTURE} is stale; see doc/fixtures/RL_OPTIONS.md"
    );
    let embedded: Value = serde_json::from_str(RL_OPTIONS_FIXTURES).unwrap();
    assert_eq!(built, embedded, "the embedded fixtures are stale");
}

#[test]
fn the_window_names_slots_exactly() {
    let p = proposed();
    let listed = listing();
    let slots = p.resolve_slots(Some(dec!(100)), &listed).unwrap();
    let name =
        |slot: usize| slots[slot].map(|i| (listed[i].expiry_symbol.as_str(), listed[i].strike));
    // Rank 0 is D1 (same instant as D1B, lower symbol), strike rank 0 the mark itself.
    assert_eq!(name(0), Some(("D1", dec!(100))));
    // Strike rank 1 at mark 100: 95 and 105 are both 5 away; ties go to the lower.
    assert_eq!(name(2), Some(("D1", dec!(95))));
    assert_eq!(name(4), Some(("D1", dec!(105))));
    // D1 105 put is not listed: the slot is empty.
    assert_eq!(name(5), None);
    // Expiry rank 1 is D1B; W1 is outside a window of two.
    assert_eq!(name(10), Some(("D1B", dec!(100))));
    assert!(
        slots
            .iter()
            .flatten()
            .all(|i| listed[*i].expiry_symbol != "W1")
    );
    assert!(
        p.resolve_slots(None, &listed)
            .unwrap()
            .iter()
            .all(Option::is_none)
    );
}

#[test]
fn refusals_come_from_the_heads_alone() {
    let p = small();
    let hedge = p.hedge_lane();
    assert!(p.decode_action(&[4, 0, 0, 0, 0]).is_err());
    assert!(p.decode_action(&[1, hedge, 0, 0, 0]).is_err());
    assert!(p.decode_action(&[2, hedge, 0, 0, 0]).is_err());
    assert!(p.decode_action(&[3, hedge, 0, 0, 0]).is_err());
    assert!(p.decode_action(&[4, hedge, 0, 0, 0]).is_ok());
    assert!(p.decode_action(&[5, hedge, 0, 0, 0]).is_ok());
    assert!(p.decode_action(&[0, hedge + 1, 0, 0, 0]).is_err());
    let a = OptionsAction::new(OptionsKind::Quote, Lane::Option(3), 1, 0, 0);
    assert_eq!(p.decode_action(&p.encode_action(&a).unwrap()).unwrap(), a);
}

#[test]
fn the_profile_refuses_a_non_positive_sigma() {
    let mut p = proposed();
    p.sigma_delta = Decimal::ZERO;
    assert!(matches!(
        p.validate(),
        Err(MMError::InvalidConfiguration(_))
    ));
    p.sigma_delta = dec!(-0.1);
    assert!(p.id().is_err());
}

#[test]
fn the_fingerprint_moves_with_every_constant() {
    type Change = (&'static str, fn(&mut OptionsProfile));
    let changes: &[Change] = &[
        ("expiry_ranks", |p| p.expiry_ranks = 3),
        ("strike_ranks", |p| p.strike_ranks = 4),
        ("sigma_delta", |p| p.sigma_delta = dec!(0.61)),
        ("scales.strike_distance", |p| {
            p.scales.strike_distance = dec!(999)
        }),
        ("scales.time_to_expiry_ms", |p| {
            p.scales.time_to_expiry_ms = dec!(1)
        }),
        ("scales.offset_ticks", |p| p.scales.offset_ticks = dec!(21)),
        ("scales.depth_lots", |p| p.scales.depth_lots = dec!(51)),
        ("scales.venue_mark", |p| p.scales.venue_mark = dec!(501)),
        ("scales.position_lots", |p| {
            p.scales.position_lots = dec!(22)
        }),
        ("scales.equity", |p| p.scales.equity = dec!(10001)),
        ("scales.net_delta", |p| p.scales.net_delta = dec!(11)),
        ("hedge_scales.spread_ticks", |p| {
            p.hedge_scales.spread_ticks = dec!(12)
        }),
        ("hedge_scales.mid_change_ticks", |p| {
            p.hedge_scales.mid_change_ticks = dec!(13)
        }),
        ("hedge_scales.row_gap_ms", |p| {
            p.hedge_scales.row_gap_ms = dec!(1001)
        }),
        ("hedge_scales.size_lots", |p| {
            p.hedge_scales.size_lots = dec!(14)
        }),
        ("hedge_scales.position", |p| {
            p.hedge_scales.position = dec!(15)
        }),
        ("hedge_scales.entry_gap", |p| {
            p.hedge_scales.entry_gap = dec!(101)
        }),
        ("bid_offsets_ticks", |p| {
            p.bid_offsets_ticks = vec![1, 2, 4, 9]
        }),
        ("ask_offsets_ticks", |p| p.ask_offsets_ticks = vec![1, 2, 4]),
        ("sizes_lots", |p| p.sizes_lots = vec![1, 2, 6]),
        ("max_orders", |p| p.max_orders = 3),
        ("max_steps", |p| p.max_steps = 18),
        ("step_ns", |p| p.step_ns = 999_999_999),
    ];
    let base = proposed().id().unwrap();
    let mut seen = vec![base];
    for (name, change) in changes {
        let mut p = proposed();
        change(&mut p);
        let id = p.id().unwrap();
        assert!(!seen.contains(&id), "{name} does not move the fingerprint");
        seen.push(id);
    }
    assert!(proposed().ensure(base).is_ok());
}

#[test]
fn put_call_parity_holds_on_every_fixture() {
    let fixtures: Value = serde_json::from_str(RL_OPTIONS_FIXTURES).unwrap();
    let rows = fixtures["deltas"].as_array().unwrap();
    for pair in rows.chunks(2) {
        let call = Decimal::from_str(pair[0][5].as_str().unwrap()).unwrap();
        let put = Decimal::from_str(pair[1][5].as_str().unwrap()).unwrap();
        let tau = Decimal::from_str(pair[0][2].as_str().unwrap()).unwrap();
        if tau > Decimal::ZERO {
            // Each side is rounded on its own, so the difference is 1 within two half units.
            assert!(
                (call - put - Decimal::ONE).abs() <= dec!(0.000000001),
                "{pair:?}"
            );
        }
    }
}
