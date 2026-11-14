//! The spaces from outside the environment: a specification from a declared profile, the
//! observation from explicit inputs, the decoding of the action heads and the published
//! fixtures in `doc/fixtures/rl_spaces.json`. All data here is synthetic.
//!
//! Run with `MM_WRITE_RL_FIXTURES=1` to rewrite the fixture file after a deliberate change
//! of version.

#![cfg(feature = "rl")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::str::FromStr;

use market_maker_rs::Decimal;
use market_maker_rs::backtest::MarketTick;
use market_maker_rs::backtest::kernel::{
    ActionGrid, KernelConfig, KernelDataset, Observation, QuoteSide, Status,
};
use market_maker_rs::dec;
use market_maker_rs::rl::{
    ACTION_HEAD_COUNT, ACTION_KIND_COUNT, ActionHead, InstrumentAction, InstrumentRole,
    InstrumentSpace, InvalidActionPolicy, MULTI_INSTRUMENT_ACTION_SPACE_VERSION, ObservationScales,
    ObservationSpace, PreviousRow, QuoteLives, RewardMark, RlAction, RlConfig, RlEnv, RlProfile,
    RlSpec, TerminalValuation, observation_features,
};
use market_maker_rs::types::error::MMError;
use serde_json::{Value, json};

const FIXTURE: &str = "doc/fixtures/rl_spaces.json";

fn scales() -> ObservationScales {
    ObservationScales {
        spread_ticks: dec!(10),
        mid_change_ticks: dec!(10),
        row_gap_ms: dec!(1000),
        size_lots: dec!(10),
        position: dec!(10),
        entry_gap: dec!(1),
    }
}

fn kernel() -> KernelConfig {
    KernelConfig::new(dec!(0.5), dec!(1), dec!(1000), dec!(10), 100).with_fee_rate(dec!(0.001))
}

fn ticks() -> Vec<MarketTick> {
    vec![
        MarketTick::new(1000, dec!(99.5), dec!(5), dec!(100.5), dec!(5)),
        MarketTick::new(2000, dec!(98.5), dec!(3), dec!(99.5), dec!(7)),
        MarketTick::new(3500, dec!(100), dec!(5), dec!(101), dec!(1)),
        MarketTick::new(4000, dec!(100), dec!(2), dec!(101.5), dec!(4)),
    ]
}

fn grid() -> ActionGrid<3, 2> {
    ActionGrid::new([0, 1, 10], [0, 1, 10], [1, 2]).unwrap()
}

fn quoted(cfg: &KernelConfig) -> InstrumentSpace {
    InstrumentSpace::new(
        InstrumentRole::Quoted,
        cfg.tick_size,
        cfg.lot_size,
        cfg.max_price,
        cfg.max_order_size,
        vec![0, 1, 10],
        vec![0, 1, 10],
        vec![1, 2],
    )
}

fn hedge() -> InstrumentSpace {
    InstrumentSpace::new(
        InstrumentRole::Hedge,
        dec!(0.1),
        dec!(0.01),
        dec!(100000),
        dec!(5),
        vec![0, 2],
        vec![0, 2],
        vec![1, 5, 10],
    )
    .with_depth(5)
}

/// The configurations an environment of this crate can state, each by name.
fn configs() -> Vec<(&'static str, RlConfig)> {
    let base = RlConfig::new(scales(), dec!(1));
    vec![
        ("plain", base),
        ("hedge_actions", base.with_hedge_actions()),
        ("cancel", base.with_cancel_action()),
        (
            "lives",
            base.with_cancel_action()
                .with_quote_lives(QuoteLives::new(&[1_000, 5_000]).unwrap()),
        ),
        (
            "reject_close",
            base.with_invalid_action(InvalidActionPolicy::Reject, dec!(0.25))
                .with_terminal(TerminalValuation::CloseAtTouch),
        ),
    ]
}

fn env_spec(config: RlConfig) -> RlSpec {
    let cfg = kernel();
    let data = KernelDataset::from_ticks(&ticks(), &cfg).unwrap();
    let env: RlEnv<'_, 3, 2> = RlEnv::new(&cfg, &data, grid(), config).unwrap();
    env.spec().clone()
}

fn profile(config: RlConfig) -> RlProfile {
    let cfg = kernel();
    RlProfile::new(vec![quoted(&cfg)], config, 16, cfg.max_steps)
}

/// The declared profiles of the fixture file.
fn fixture_profiles() -> Vec<(&'static str, RlProfile)> {
    let cfg = kernel();
    let base = RlConfig::new(scales(), dec!(1));
    let mut out: Vec<(&'static str, RlProfile)> = configs()
        .into_iter()
        .map(|(name, config)| (name, profile(config)))
        .collect();
    out.push((
        "external_mark",
        profile(base).with_reward_mark(RewardMark::ExternalVenueMark),
    ));
    out.push((
        "two_instruments",
        RlProfile::new(
            vec![quoted(&cfg).with_max_orders(3), hedge()],
            base.with_hedge_actions().with_cancel_action(),
            32,
            500,
        )
        .with_reward_mark(RewardMark::ExternalVenueMark),
    ));
    out
}

#[test]
fn test_profile_spec_equals_the_environment_spec_to_the_byte() {
    for (name, config) in configs() {
        let from_env = env_spec(config);
        let from_profile = profile(config).spec().unwrap();
        assert_eq!(from_profile, from_env, "{name}");
        assert_eq!(from_profile.to_bytes(), from_env.to_bytes(), "{name}");
        assert_eq!(from_profile.id(), from_env.id(), "{name}");
        assert!(from_profile.action.instruments.is_empty(), "{name}");
    }
}

#[test]
fn test_external_mark_and_instruments_change_the_identity() {
    let base = RlConfig::new(scales(), dec!(1));
    let plain = profile(base).spec().unwrap();
    let marked = profile(base)
        .with_reward_mark(RewardMark::ExternalVenueMark)
        .spec()
        .unwrap();
    assert_eq!(marked.reward.mark, RewardMark::ExternalVenueMark);
    assert_eq!(marked.action.version, plain.action.version);
    assert_ne!(marked.id().fingerprint, plain.id().fingerprint);
    assert_ne!(marked.to_bytes(), plain.to_bytes());

    let cfg = kernel();
    let deeper = RlProfile::new(vec![quoted(&cfg).with_depth(2)], base, 16, cfg.max_steps)
        .spec()
        .unwrap();
    assert_eq!(deeper.action.version, MULTI_INSTRUMENT_ACTION_SPACE_VERSION);
    assert_eq!(deeper.action.instruments.len(), 1);
    let more = RlProfile::new(
        vec![quoted(&cfg).with_max_orders(2)],
        base,
        16,
        cfg.max_steps,
    )
    .spec()
    .unwrap();
    let two = RlProfile::new(vec![quoted(&cfg), hedge()], base, 16, cfg.max_steps)
        .spec()
        .unwrap();
    let swapped_role = RlProfile::new(
        vec![quoted(&cfg), {
            let mut h = hedge();
            h.role = InstrumentRole::Quoted;
            h
        }],
        base,
        16,
        cfg.max_steps,
    )
    .spec()
    .unwrap();
    let ids = [
        plain.id().fingerprint,
        deeper.id().fingerprint,
        more.id().fingerprint,
        two.id().fingerprint,
        swapped_role.id().fingerprint,
    ];
    for (i, a) in ids.iter().enumerate() {
        for b in ids.iter().skip(i + 1) {
            assert_ne!(a, b);
        }
    }
    assert_eq!(two.action.instruments[0], quoted(&cfg));
    assert_eq!(
        two.action_head_sizes(),
        [ACTION_KIND_COUNT, 2, 3, 3, 3, 3, 1]
    );
}

#[test]
fn test_profile_refuses_what_it_cannot_state() {
    let cfg = kernel();
    let base = RlConfig::new(scales(), dec!(1));
    let refused = |p: RlProfile| matches!(p.spec(), Err(MMError::InvalidConfiguration(_)));
    assert!(refused(RlProfile::new(vec![], base, 16, 100)));
    assert!(refused(RlProfile::new(vec![hedge()], base, 16, 100)));
    assert!(refused(RlProfile::new(vec![quoted(&cfg)], base, 0, 100)));
    assert!(refused(RlProfile::new(vec![quoted(&cfg)], base, 16, 0)));
    assert!(refused(RlProfile::new(
        vec![quoted(&cfg)],
        RlConfig::new(scales(), dec!(0)),
        16,
        100
    )));
    let mut bad_scales = scales();
    bad_scales.position = dec!(0);
    assert!(refused(RlProfile::new(
        vec![quoted(&cfg)],
        RlConfig::new(bad_scales, dec!(1)),
        16,
        100
    )));
    assert!(refused(RlProfile::new(
        vec![quoted(&cfg)],
        base.with_invalid_action(InvalidActionPolicy::Reject, dec!(-1)),
        16,
        100
    )));
    for broken in [
        quoted(&cfg).with_depth(0),
        quoted(&cfg).with_max_orders(0),
        {
            let mut i = quoted(&cfg);
            i.tick_size = dec!(0);
            i
        },
        {
            let mut i = quoted(&cfg);
            i.sizes_lots = vec![1, 0];
            i
        },
        {
            let mut i = quoted(&cfg);
            i.ask_offsets_ticks = vec![0];
            i
        },
        {
            let mut i = quoted(&cfg);
            i.bid_offsets_ticks = vec![];
            i.ask_offsets_ticks = vec![];
            i
        },
    ] {
        assert!(refused(RlProfile::new(vec![broken], base, 16, 100)));
    }
    let many = vec![quoted(&cfg); 65];
    assert!(refused(RlProfile::new(many, base, 16, 100)));
}

#[test]
fn test_observation_from_explicit_inputs_equals_the_environment() {
    let cfg = kernel();
    let data = KernelDataset::from_ticks(&ticks(), &cfg).unwrap();
    for (name, config) in configs() {
        let env: RlEnv<'_, 3, 2> = RlEnv::new(&cfg, &data, grid(), config).unwrap();
        let spec = profile(config).spec().unwrap();
        let space = ObservationSpace::from_scales(&config.scales, 16, cfg.max_steps).unwrap();
        assert_eq!(space, spec.observation);
        let mut state = env.reset(3).unwrap();
        let mut previous: Option<PreviousRow> = None;
        let actions = [
            RlAction::Quote {
                bid: 0,
                ask: 0,
                bid_size: 1,
                ask_size: 0,
            },
            RlAction::Quote {
                bid: 1,
                ask: 0,
                bid_size: 0,
                ask_size: 1,
            },
            RlAction::Hold,
            RlAction::Hold,
            RlAction::Hold,
        ];
        let mut steps = 0;
        for action in actions {
            if state.is_done() {
                break;
            }
            let obs = env.kernel().observe(state.kernel());
            let ours =
                observation_features(&space, &obs, previous, cfg.tick_size, cfg.max_steps).unwrap();
            assert_eq!(ours, env.observe(&state).unwrap(), "{name} step {steps}");
            previous = PreviousRow::of(&obs).or(previous);
            env.step(&mut state, action).unwrap();
            steps += 1;
        }
        assert!(steps >= 3, "{name}");
    }
}

#[test]
fn test_action_heads_round_trip() {
    let cfg = kernel();
    let base = RlConfig::new(scales(), dec!(1));
    let lives = base.with_quote_lives(QuoteLives::new(&[10, 20, 30]).unwrap());
    for spec in [
        profile(base).spec().unwrap(),
        profile(lives).spec().unwrap(),
        RlProfile::new(vec![quoted(&cfg), hedge()], lives, 16, 100)
            .spec()
            .unwrap(),
    ] {
        let sizes = spec.action_head_sizes();
        assert_eq!(sizes.len(), ACTION_HEAD_COUNT);
        let mut decoded = 0;
        // Every row of the head sizes either decodes to an action that encodes back to its
        // canonical row, or is out of range for its instrument.
        let total: usize = sizes.iter().product();
        for mut n in 0..total {
            let mut row = [0; ACTION_HEAD_COUNT];
            for (slot, size) in row.iter_mut().zip(sizes) {
                *slot = n % size;
                n /= size;
            }
            match spec.decode_action(&row) {
                Ok(action) => {
                    decoded += 1;
                    let canonical = spec.encode_action(&action).unwrap();
                    assert_eq!(spec.decode_action(&canonical).unwrap(), action);
                    for head in ActionHead::ALL {
                        let i = head.index();
                        assert!(canonical[i] == row[i] || canonical[i] == 0, "{row:?}");
                    }
                }
                Err(e) => assert!(matches!(e, MMError::InvalidConfiguration(_))),
            }
        }
        assert!(decoded > 0);
        assert!(spec.decode_action(&[0; 6]).is_err());
        assert!(
            spec.decode_action(&[ACTION_KIND_COUNT, 0, 0, 0, 0, 0, 0])
                .is_err()
        );
        assert!(spec.decode_action(&[0, sizes[1], 0, 0, 0, 0, 0]).is_err());
        // No hedge lane, so no hedge size in the second instrument.
        assert!(spec.decode_action(&[5, 0, 0, 0, 0, 0, 0]).is_err());
    }
    // Without lives an expiring quote has no life to point at.
    let plain = profile(base).spec().unwrap();
    assert!(plain.decode_action(&[7, 0, 0, 0, 0, 0, 0]).is_err());
    assert_eq!(
        plain.decode_action(&[3, 0, 9, 2, 1, 1, 9]).unwrap(),
        InstrumentAction::new(
            0,
            RlAction::OneSided {
                side: QuoteSide::Ask,
                offset: 2,
                size: 1
            }
        )
    );
}

// ---------------------------------------------------------------- fixtures

fn d(value: Decimal) -> Value {
    json!(value.normalize().to_string())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn role_name(role: InstrumentRole) -> &'static str {
    match role {
        InstrumentRole::Quoted => "quoted",
        _ => "hedge",
    }
}

fn mark_name(mark: RewardMark) -> &'static str {
    match mark {
        RewardMark::LastCompletedRow => "last_completed_row",
        _ => "external_venue_mark",
    }
}

fn profile_json(p: &RlProfile) -> Value {
    let c = &p.config;
    json!({
        "instruments": p.instruments.iter().map(|i| json!({
            "role": role_name(i.role),
            "tick_size": d(i.tick_size),
            "lot_size": d(i.lot_size),
            "max_price": d(i.max_price),
            "max_order_size": d(i.max_order_size),
            "depth": i.depth,
            "bid_offsets_ticks": i.bid_offsets_ticks,
            "ask_offsets_ticks": i.ask_offsets_ticks,
            "sizes_lots": i.sizes_lots,
            "max_orders": i.max_orders,
        })).collect::<Vec<_>>(),
        "scales": {
            "spread_ticks": d(c.scales.spread_ticks),
            "mid_change_ticks": d(c.scales.mid_change_ticks),
            "row_gap_ms": d(c.scales.row_gap_ms),
            "size_lots": d(c.scales.size_lots),
            "position": d(c.scales.position),
            "entry_gap": d(c.scales.entry_gap),
        },
        "f32_unit": d(c.f32_unit),
        "invalid_action": match c.invalid_action {
            InvalidActionPolicy::Mask => "mask",
            _ => "reject",
        },
        "invalid_action_penalty": d(c.invalid_action_penalty),
        "terminal": match c.terminal {
            TerminalValuation::MarkAtLastMid => "mark_at_last_mid",
            _ => "close_at_touch",
        },
        "hedge_actions": c.hedge_actions,
        "cancel_action": c.cancel_action,
        "quote_lives_ns": c.quote_lives.map(|l| l.as_slice().to_vec()),
        "order_capacity": p.order_capacity,
        "max_steps": p.max_steps,
        "reward_mark": mark_name(p.reward_mark),
    })
}

/// The profile of a fixture entry, read back as an external environment would.
fn profile_from_json(v: &Value) -> RlProfile {
    let dec_of = |v: &Value| Decimal::from_str(v.as_str().unwrap()).unwrap();
    let u = |v: &Value| v.as_u64().unwrap();
    let instruments = v["instruments"]
        .as_array()
        .unwrap()
        .iter()
        .map(|i| {
            InstrumentSpace::new(
                if i["role"] == "quoted" {
                    InstrumentRole::Quoted
                } else {
                    InstrumentRole::Hedge
                },
                dec_of(&i["tick_size"]),
                dec_of(&i["lot_size"]),
                dec_of(&i["max_price"]),
                dec_of(&i["max_order_size"]),
                serde_json::from_value(i["bid_offsets_ticks"].clone()).unwrap(),
                serde_json::from_value(i["ask_offsets_ticks"].clone()).unwrap(),
                serde_json::from_value(i["sizes_lots"].clone()).unwrap(),
            )
            .with_depth(u16::try_from(u(&i["depth"])).unwrap())
            .with_max_orders(u16::try_from(u(&i["max_orders"])).unwrap())
        })
        .collect();
    let s = &v["scales"];
    let mut config = RlConfig::new(
        ObservationScales {
            spread_ticks: dec_of(&s["spread_ticks"]),
            mid_change_ticks: dec_of(&s["mid_change_ticks"]),
            row_gap_ms: dec_of(&s["row_gap_ms"]),
            size_lots: dec_of(&s["size_lots"]),
            position: dec_of(&s["position"]),
            entry_gap: dec_of(&s["entry_gap"]),
        },
        dec_of(&v["f32_unit"]),
    )
    .with_invalid_action(
        if v["invalid_action"] == "mask" {
            InvalidActionPolicy::Mask
        } else {
            InvalidActionPolicy::Reject
        },
        dec_of(&v["invalid_action_penalty"]),
    )
    .with_terminal(if v["terminal"] == "mark_at_last_mid" {
        TerminalValuation::MarkAtLastMid
    } else {
        TerminalValuation::CloseAtTouch
    });
    if v["hedge_actions"].as_bool().unwrap() {
        config = config.with_hedge_actions();
    }
    if v["cancel_action"].as_bool().unwrap() {
        config = config.with_cancel_action();
    }
    if let Some(lives) = v["quote_lives_ns"].as_array() {
        let lives: Vec<u64> = lives.iter().map(u).collect();
        config = config.with_quote_lives(QuoteLives::new(&lives).unwrap());
    }
    RlProfile::new(
        instruments,
        config,
        u(&v["order_capacity"]),
        u(&v["max_steps"]),
    )
    .with_reward_mark(if v["reward_mark"] == "last_completed_row" {
        RewardMark::LastCompletedRow
    } else {
        RewardMark::ExternalVenueMark
    })
}

fn action_json(a: &InstrumentAction) -> Value {
    let side_name = |s: QuoteSide| match s {
        QuoteSide::Bid => "bid",
        _ => "ask",
    };
    let action = match a.action {
        RlAction::Hold => json!({"kind": "hold"}),
        RlAction::Quote {
            bid,
            ask,
            bid_size,
            ask_size,
        } => {
            json!({"kind": "quote", "bid": bid, "ask": ask, "bid_size": bid_size, "ask_size": ask_size})
        }
        RlAction::OneSided { side, offset, size } => {
            json!({"kind": "one_sided", "side": side_name(side), "offset": offset, "size": size})
        }
        RlAction::Hedge { size } => json!({"kind": "hedge", "size": size}),
        RlAction::CrossHedge { size } => json!({"kind": "cross_hedge", "size": size}),
        RlAction::Cancel => json!({"kind": "cancel"}),
        RlAction::ExpiringQuote {
            bid,
            ask,
            bid_size,
            ask_size,
            life,
        } => {
            json!({"kind": "expiring_quote", "bid": bid, "ask": ask, "bid_size": bid_size, "ask_size": ask_size, "life": life})
        }
        _ => panic!("an action the fixtures do not know"),
    };
    json!({"instrument": a.instrument, "action": action})
}

fn obs(
    row: Option<u64>,
    ts_ns: u64,
    book: (u64, u64, u64, u64),
    position: Decimal,
    avg_entry: Decimal,
    orders: (u64, u64),
    step: u64,
) -> Observation {
    Observation {
        row,
        ts_ns,
        bid: book.0,
        bid_lots: book.1,
        ask: book.2,
        ask_lots: book.3,
        position,
        avg_entry,
        delivered_fills: 0,
        delivered_local_rejects: 0,
        open_orders: orders.0,
        awaiting_cancel_outcome: orders.1,
        step,
        status: Status::Running,
        rejected: [0; 8],
    }
}

/// Plain inputs, each with what the previous decision left.
fn observation_cases() -> Vec<(&'static str, Observation, Option<PreviousRow>)> {
    vec![
        (
            "no_row_yet",
            obs(None, 0, (0, 0, 0, 0), dec!(0), dec!(0), (0, 0), 0),
            None,
        ),
        (
            "first_row_flat",
            obs(
                Some(0),
                1_000,
                (199, 5, 201, 5),
                dec!(0),
                dec!(0),
                (0, 0),
                0,
            ),
            None,
        ),
        (
            "long_after_a_move",
            obs(
                Some(1),
                2_500_000,
                (197, 3, 199, 7),
                dec!(2),
                dec!(99.75),
                (2, 1),
                1,
            ),
            Some(PreviousRow::new(400, 1_000)),
        ),
        (
            "short_clipped",
            obs(
                Some(7),
                9_000_000_000,
                (300, 40, 340, 0),
                dec!(-25),
                dec!(150.5),
                (20, 0),
                99,
            ),
            Some(PreviousRow::new(398, 2_500_000)),
        ),
    ]
}

/// Head rows for the fixture, from the head sizes of `spec`: every kind on the last
/// instrument with the last value of each head, the first values, and some rows out of range.
fn action_rows(spec: &RlSpec) -> Vec<Vec<usize>> {
    let sizes = spec.action_head_sizes();
    let last: Vec<usize> = sizes.iter().map(|s| s - 1).collect();
    let mut rows = Vec::new();
    for kind in 0..ACTION_KIND_COUNT {
        let mut first = vec![0; ACTION_HEAD_COUNT];
        first[0] = kind;
        rows.push(first);
        let mut high = last.clone();
        high[0] = kind;
        rows.push(high);
    }
    rows.push(vec![ACTION_KIND_COUNT, 0, 0, 0, 0, 0, 0]);
    rows.push(vec![1, sizes[1], 0, 0, 0, 0, 0]);
    rows.push(vec![1, 0, sizes[2], 0, 0, 0, 0]);
    rows
}

fn fixtures() -> Value {
    let mut profiles = Vec::new();
    let mut observations = Vec::new();
    let mut actions = Vec::new();
    for (name, p) in fixture_profiles() {
        let spec = p.spec().unwrap();
        let id = spec.id();
        profiles.push(json!({
            "name": name,
            "profile": profile_json(&p),
            "spec_bytes_hex": hex(&spec.to_bytes()),
            "spec_id": {
                "observation": id.observation,
                "action": id.action,
                "reward": id.reward,
                "episode": id.episode,
                "fingerprint": id.fingerprint.to_string(),
            },
            "action_head_sizes": spec.action_head_sizes(),
        }));
        for row in action_rows(&spec) {
            let expected = match spec.decode_action(&row) {
                Ok(action) => {
                    let canonical = spec.encode_action(&action).unwrap();
                    json!({"decoded": action_json(&action), "canonical_heads": canonical})
                }
                Err(_) => json!({"error": "invalid_configuration"}),
            };
            actions.push(json!({"profile": name, "heads": row, "expected": expected}));
        }
        if name != "plain" && name != "two_instruments" {
            continue;
        }
        let tick = p.instruments[0].tick_size;
        for (case, o, previous) in observation_cases() {
            let v =
                observation_features(&spec.observation, &o, previous, tick, p.max_steps).unwrap();
            observations.push(json!({
                "profile": name,
                "case": case,
                "tick_size": d(tick),
                "previous": previous.map(|r| json!({"mid2": r.mid2, "ts_ns": r.ts_ns})),
                "observation": {
                    "row": o.row,
                    "ts_ns": o.ts_ns,
                    "bid": o.bid,
                    "bid_lots": o.bid_lots,
                    "ask": o.ask,
                    "ask_lots": o.ask_lots,
                    "position": d(o.position),
                    "avg_entry": d(o.avg_entry),
                    "open_orders": o.open_orders,
                    "awaiting_cancel_outcome": o.awaiting_cancel_outcome,
                    "step": o.step,
                },
                "expected_raw": v.raw.iter().map(|r| r.map(d)).collect::<Vec<_>>(),
                "expected_values": v.values.iter().map(|x| format!("{x:?}")).collect::<Vec<_>>(),
            }));
        }
    }
    json!({
        "format": 1,
        "about": "Fixtures of the rl spaces: declared profiles with their spec bytes and ids, plain observation inputs with their expected feature vectors, and action head rows with their expected decoded actions. Decimals and f32 values are strings, the f32 in its shortest round trip form. Generated by tests/rl_external.rs.",
        "profiles": profiles,
        "observations": observations,
        "actions": actions,
    })
}

#[test]
fn test_published_fixtures_match() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURE);
    let generated = fixtures();
    if std::env::var_os("MM_WRITE_RL_FIXTURES").is_some() {
        let mut text = serde_json::to_string_pretty(&generated).unwrap();
        text.push('\n');
        std::fs::write(&path, text).unwrap();
    }
    let published: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    assert_eq!(published, generated, "{FIXTURE} is out of date");
    rebuild_from_plain_data(&published);
}

/// What an external environment does with the file, after the comparison so that a rewrite
/// comes first: rebuild each profile from its plain
/// data, and compare bytes, ids and vectors.
fn rebuild_from_plain_data(published: &Value) {
    let mut specs = std::collections::HashMap::new();
    for entry in published["profiles"].as_array().unwrap() {
        let p = profile_from_json(&entry["profile"]);
        let spec = p.spec().unwrap();
        assert_eq!(hex(&spec.to_bytes()), entry["spec_bytes_hex"]);
        assert_eq!(
            spec.id().fingerprint.to_string(),
            entry["spec_id"]["fingerprint"]
        );
        specs.insert(entry["name"].as_str().unwrap().to_string(), (p, spec));
    }
    let plain = &specs["plain"].1;
    assert_eq!(plain, &env_spec(RlConfig::new(scales(), dec!(1))));
    for entry in published["observations"].as_array().unwrap() {
        let (p, spec) = &specs[entry["profile"].as_str().unwrap()];
        let o = &entry["observation"];
        let u = |v: &Value| v.as_u64().unwrap();
        let observation = obs(
            o["row"].as_u64(),
            u(&o["ts_ns"]),
            (
                u(&o["bid"]),
                u(&o["bid_lots"]),
                u(&o["ask"]),
                u(&o["ask_lots"]),
            ),
            Decimal::from_str(o["position"].as_str().unwrap()).unwrap(),
            Decimal::from_str(o["avg_entry"].as_str().unwrap()).unwrap(),
            (u(&o["open_orders"]), u(&o["awaiting_cancel_outcome"])),
            u(&o["step"]),
        );
        let previous = entry["previous"]
            .as_object()
            .map(|r| PreviousRow::new(u(&r["mid2"]), u(&r["ts_ns"])));
        let tick = Decimal::from_str(entry["tick_size"].as_str().unwrap()).unwrap();
        let v = observation_features(&spec.observation, &observation, previous, tick, p.max_steps)
            .unwrap();
        for (i, raw) in v.raw.iter().enumerate() {
            let expected = &entry["expected_raw"][i];
            match raw {
                Some(raw) => {
                    assert_eq!(*raw, Decimal::from_str(expected.as_str().unwrap()).unwrap())
                }
                None => assert!(expected.is_null()),
            }
            let value: f32 = entry["expected_values"][i]
                .as_str()
                .unwrap()
                .parse()
                .unwrap();
            assert_eq!(v.values[i].to_bits(), value.to_bits());
        }
    }
}
