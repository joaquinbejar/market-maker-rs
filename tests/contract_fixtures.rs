//! Hand computed cashflow fixtures for the financial contract proposal (#78).
//!
//! Each fixture states its instrument, units and expected cashflows as
//! literals computed by hand (shown in the comments), and then checks them
//! against an independent property rather than re-deriving the same formula:
//! counterparty conservation, equivalence between instrument forms, sign
//! symmetry or parity. The hand fixtures depend on nothing beyond `Decimal`;
//! separate tests cross-check some of them against the published formulas
//! of `financial_types`, on its dated venue snapshots.
//!
//! Conventions used throughout:
//! - prices and money are exact `Decimal`;
//! - timestamps are milliseconds since the Unix epoch (`u64`);
//! - quantities are positive and the side carries the sign;
//! - the only rounding step is explicit and named (`round_dp_with_strategy`
//!   half even, at the settlement asset scale);
//! - arithmetic that could overflow uses checked operations.

use financial_types::Side;
use financial_types::instrument::reference::{binance_usdm, deribit};
use financial_types::instrument::{
    AssetCode, InstrumentId, Money, MoneyRatio, Ratio, RoundingRule,
};
use market_maker_rs::Decimal;
use market_maker_rs::dec;
use rust_decimal::RoundingStrategy;

/// Settlement scale of BTC (satoshi).
const BTC_SCALE: u32 = 8;

fn round_btc(value: Decimal) -> Decimal {
    value.round_dp_with_strategy(BTC_SCALE, RoundingStrategy::MidpointNearestEven)
}

// ---------------------------------------------------------------- spot

/// Spot BTC-USDT, base BTC, quote and settlement USDT, fee 0.1% of notional
/// in quote.
///
/// Buy 0.5 @ 60000: notional 30000, fee 30, cash -30030.
/// Sell 0.5 @ 61000: notional 30500, fee 30.5, cash +30469.5.
/// Net cash 439.5 = gross 500 - fees 60.5; position flat.
#[test]
fn test_spot_round_trip_net_cash_equals_gross_minus_fees() {
    let buy_cash = dec!(-30030);
    let sell_cash = dec!(30469.5);
    let fees = dec!(60.5);
    let net = buy_cash + sell_cash;
    assert_eq!(net, dec!(439.5));
    // Independent check: gross PnL is quantity times the price move.
    let gross = dec!(0.5) * (dec!(61000) - dec!(60000));
    assert_eq!(net + fees, gross);
}

/// Conservation: between the two sides of a spot trade, cash and base move
/// in opposite directions and only fees leave the pair.
///
/// Trade 0.5 BTC @ 60000, each side pays 0.1%: buyer cash -30030, seller
/// cash +29970, base +0.5 and -0.5.
#[test]
fn test_spot_trade_counterparties_conserve_value_except_fees() {
    let (buyer_cash, seller_cash) = (dec!(-30030), dec!(29970));
    let (buyer_base, seller_base) = (dec!(0.5), dec!(-0.5));
    assert_eq!(buyer_base + seller_base, Decimal::ZERO);
    assert_eq!(buyer_cash + seller_cash, -dec!(60));
}

// ------------------------------------------------------ linear future

/// Linear (USDT margined) BTC contract as listed by Binance (`BTCUSDT`):
/// multiplier 1 BTC per contract, lot step 0.001 BTC, quote and settlement
/// USDT.
///
/// Long 0.01 contracts (10 lot steps) @ 60000, close @ 61000: PnL 10 USDT.
/// Short the same: -10 USDT.
#[test]
fn test_linear_future_close_pnl_matches_spot_exposure() {
    let long_pnl = dec!(10);
    let short_pnl = dec!(-10);
    // Independent check: 0.01 contracts x 1 BTC is 0.01 BTC of exposure,
    // and a 0.01 BTC spot position gains the same over the same move.
    let exposure_btc = dec!(0.01);
    let spot_gain = exposure_btc * dec!(1000);
    assert_eq!(long_pnl, spot_gain);
    assert_eq!(long_pnl + short_pnl, Decimal::ZERO);
}

/// Cross-check of the linear literals against the published formula of
/// `financial_types` on the dated Binance snapshot. Kept apart from the
/// hand fixture, so a disagreement shows which side moved.
#[test]
fn test_linear_future_literals_match_financial_types() {
    let spec = binance_usdm::btcusdt_perpetual().unwrap();
    assert_eq!(spec.size().amount(), dec!(1));
    assert_eq!(spec.lot(), dec!(0.001));
    let pnl = |side| {
        spec.realized_pnl(side, dec!(0.01), dec!(60000), dec!(61000))
            .unwrap()
    };
    let usdt = AssetCode::new("USDT").unwrap();
    let long = pnl(Side::Long);
    assert!(
        long.equals(&MoneyRatio::new(
            Ratio::from_decimal(dec!(10)),
            usdt.clone()
        ))
        .unwrap()
    );
    let short = pnl(Side::Short);
    assert!(
        short
            .equals(&MoneyRatio::new(Ratio::from_decimal(dec!(-10)), usdt))
            .unwrap()
    );
}

// ----------------------------------------------------- inverse future

/// Inverse (coin margined) BTC-USD future as listed by Deribit: contract
/// value 10 USD, settled in BTC. Specification only: margin-rs 0.1 books
/// inverse contracts, but market-maker-rs does not use them yet. PnL in BTC
/// is N x CV x (1/entry - 1/exit).
///
/// Long 100 contracts (1000 USD) @ 60000, exit 61000:
/// 1000 x (1/60000 - 1/61000) = 1000 x 1000 / 3 660 000 000 = 1/3660 BTC =
/// 0.000273224043... BTC, settled at satoshi scale (half even) as
/// 0.00027322 BTC.
/// Long 100 @ 60000, exit 59000: 1000 x (1/60000 - 1/59000) =
/// -1000 x 1000 / 3 540 000 000 = -1/3540 BTC = -0.000282485875... BTC,
/// settled as -0.00028249 BTC.
#[test]
fn test_inverse_future_close_settles_in_coin_at_satoshi_scale() {
    let up = dec!(0.00027322);
    let down = dec!(-0.00028249);

    // Independent check 1: the exact fractions, in integers. N x CV x
    // (exit - entry) over entry x exit reduces to 1/3660 and -1/3540.
    let (contracts, cv): (i128, i128) = (100, 10);
    let up_num = contracts * cv * (61_000 - 60_000);
    let up_den: i128 = 60_000 * 61_000;
    assert_eq!(up_num * 3660, up_den);
    let down_num = contracts * cv * (59_000 - 60_000);
    let down_den: i128 = 60_000 * 59_000;
    assert_eq!(down_num * 3540, -down_den);

    // Independent check 2: dividing 1 by 3660 and 3540 and rounding once
    // gives the settlements.
    let up_exact = Decimal::ONE.checked_div(dec!(3660)).unwrap();
    let down_exact = -Decimal::ONE.checked_div(dec!(3540)).unwrap();
    assert_eq!(round_btc(up_exact), up);
    assert_eq!(round_btc(down_exact), down);

    // Independent check 3: the USD value of the coin PnL at the exit price
    // equals the USD notional times the relative move against entry:
    // up: 1000 x (61000/60000 - 1) = 16.666... USD.
    let usd_at_exit = (up_exact * dec!(61000)).round_dp(6);
    let relative_move = (dec!(1000) * (dec!(61000) / dec!(60000) - Decimal::ONE)).round_dp(6);
    assert_eq!(usd_at_exit, relative_move);
    assert_eq!(usd_at_exit, dec!(16.666667));

    // Convexity: an equal move up gains less coin than the move down loses.
    assert!(up.abs() < down.abs());
}

/// Cross-check of the inverse literals against the published formulas of
/// `financial_types` (`realized_pnl`, then `settle` at satoshi scale, half
/// even) on the dated Deribit snapshot. Kept apart from the hand fixture,
/// so a disagreement shows which side moved.
#[test]
fn test_inverse_future_literals_match_financial_types() {
    let spec =
        deribit::btc_future(InstrumentId::new("BTC-27NOV26").unwrap(), 1_795_766_400_000).unwrap();
    assert_eq!(spec.size().amount(), dec!(10));
    let btc = AssetCode::new("BTC").unwrap();
    for (exit, (num, den), settled) in [
        (dec!(61000), (dec!(1), dec!(3660)), dec!(0.00027322)),
        (dec!(59000), (dec!(-1), dec!(3540)), dec!(-0.00028249)),
    ] {
        let pnl = spec
            .realized_pnl(Side::Long, dec!(100), dec!(60000), exit)
            .unwrap();
        let exact = MoneyRatio::new(Ratio::new(num, den).unwrap(), btc.clone());
        assert!(pnl.equals(&exact).unwrap(), "exact pnl at exit {exit}");
        let rounded = pnl.settle(BTC_SCALE, RoundingRule::HalfEven).unwrap();
        assert_eq!(rounded.amount, Money::new(settled, btc.clone()));
    }
}

// ---------------------------------------------------------- perpetual

/// Linear BTC-USDT perpetual, funding every 8 hours on mark notional,
/// longs pay positive rates. Long 2 BTC.
///
/// | interval | mark  | rate     | long pays |
/// | 1        | 60000 | 0.0001   | 12        |
/// | 2        | 59000 | -0.00005 | -5.9      |
/// | 3        | 61000 | 0.0002   | 24.4      |
/// Total paid by the long: 30.5 USDT; received by a 2 BTC short: 30.5.
#[test]
fn test_perpetual_funding_schedule_payments_are_zero_sum() {
    let long_payments = [dec!(12), dec!(-5.9), dec!(24.4)];
    let long_total: Decimal = long_payments.iter().sum();
    assert_eq!(long_total, dec!(30.5));
    // Independent check: each row is size x mark x rate, so the hand table
    // must agree with the schedule inputs.
    let schedule = [
        (dec!(60000), dec!(0.0001)),
        (dec!(59000), dec!(-0.00005)),
        (dec!(61000), dec!(0.0002)),
    ];
    for ((mark, rate), paid) in schedule.iter().zip(long_payments) {
        assert_eq!(dec!(2) * mark * rate, paid);
    }
    // Funding moves cash between sides; it creates or destroys none. The
    // short side is computed from its own inputs (size -2) and must net
    // against the long side.
    let short_total: Decimal = schedule
        .iter()
        .map(|(mark, rate)| dec!(-2) * mark * rate)
        .sum();
    assert_eq!(long_total + short_total, Decimal::ZERO);
    // A negative rate reverses the direction of the payment.
    assert!(long_payments[1] < Decimal::ZERO);
}

/// Funding instants are on the 8 hour grid in milliseconds, and the
/// instants a position pays are the half open interval `(opened, closed]`:
/// a position opened exactly on a funding instant does not pay it.
///
/// Hand values: the grid step is 28 800 000 ms. Opened at 1 699 977 600 000
/// (exactly on the grid: 59 027 x 28 800 000), the first instant paid is the
/// next one, 1 700 006 400 000. Opened one millisecond later, it is the same.
#[test]
fn test_perpetual_funding_schedule_paid_instants_are_half_open() {
    let interval_ms: u64 = 8 * 60 * 60 * 1000;
    assert_eq!(interval_ms, 28_800_000);
    let first_paid = |opened_at: u64| (opened_at / interval_ms + 1) * interval_ms;

    let on_grid: u64 = 1_699_977_600_000;
    assert_eq!(on_grid % interval_ms, 0);
    assert_eq!(first_paid(on_grid), 1_700_006_400_000);
    assert_eq!(first_paid(on_grid + 1), 1_700_006_400_000);
    // Opened one millisecond before an instant, that instant is paid.
    assert_eq!(first_paid(on_grid - 1), on_grid);
}

/// Funding settlement rounding: zero sum holds on exact amounts; after
/// rounding each side once at the settlement scale (2 dp, half even) a
/// residual can appear, which the ledger must absorb explicitly.
///
/// Mark 61234.57, rate 0.000123. A 0.3 BTC long pays exactly 2.259555633,
/// settled as 2.26. Three 0.1 BTC shorts each receive exactly 0.753185211,
/// settled as 0.75, so 2.25 in total. Residual: 0.01.
#[test]
fn test_perpetual_funding_rounding_leaves_explicit_residual() {
    let long_exact = dec!(2.259555633);
    let short_exact = dec!(0.753185211);
    // Exact amounts are zero sum.
    assert_eq!(long_exact - dec!(3) * short_exact, Decimal::ZERO);
    let settle = |v: Decimal| v.round_dp_with_strategy(2, RoundingStrategy::MidpointNearestEven);
    let long_paid = settle(long_exact);
    let shorts_received = dec!(3) * settle(short_exact);
    assert_eq!(long_paid, dec!(2.26));
    assert_eq!(shorts_received, dec!(2.25));
    // Rounded once per side, the residual is explicit and small.
    assert_eq!(long_paid - shorts_received, dec!(0.01));
}

// -------------------------------------------------------------- option

/// European BTC call, strike 60000, premium 1500 USD per BTC, cash settled
/// in USD at expiry, quantity 2 BTC.
///
/// Expiry 63000: payoff 2 x 3000 = 6000; buyer net 6000 - 3000 = 3000.
/// Expiry 59000: payoff 0; buyer net -3000. Seller is the mirror image.
#[test]
fn test_option_expiry_cashflows_mirror_between_sides() {
    let premium_paid = dec!(3000);
    let itm_payoff = dec!(6000);
    let otm_payoff = Decimal::ZERO;
    let buyer_itm = itm_payoff - premium_paid;
    let buyer_otm = otm_payoff - premium_paid;
    assert_eq!(buyer_itm, dec!(3000));
    assert_eq!(buyer_otm, dec!(-3000));
    // Zero sum between buyer and seller.
    let seller_itm = premium_paid - itm_payoff;
    assert_eq!(buyer_itm + seller_itm, Decimal::ZERO);
}

/// Put-call parity at expiry: a long call and a short put with the same
/// strike pay exactly the forward payoff S - K, whatever S is.
///
/// Call K=60000 payoffs at S=63000 and 59000: 3000 and 0.
/// Put K=60000 payoffs at the same S: 0 and 1000.
#[test]
fn test_option_payoffs_at_expiry_satisfy_parity() {
    let cases = [
        (dec!(63000), dec!(3000), Decimal::ZERO),
        (dec!(59000), Decimal::ZERO, dec!(1000)),
    ];
    for (spot, call, put) in cases {
        assert_eq!(call - put, spot - dec!(60000));
    }
}

// ----------------------------------------------- rounding and overflow

/// Overflow is an explicit outcome, never a panic: a notional that does not
/// fit is reported as `None` by checked arithmetic.
#[test]
fn test_notional_overflowing_product_is_detected() {
    assert_eq!(Decimal::MAX.checked_mul(dec!(2)), None);
    assert_eq!(dec!(60000).checked_mul(dec!(0.5)), Some(dec!(30000)));
}

/// Half even rounding at satoshi scale is unbiased on exact midpoints.
#[test]
fn test_settlement_rounding_midpoint_rounds_half_even() {
    assert_eq!(round_btc(dec!(0.000000025)), dec!(0.00000002));
    assert_eq!(round_btc(dec!(0.000000035)), dec!(0.00000004));
}
