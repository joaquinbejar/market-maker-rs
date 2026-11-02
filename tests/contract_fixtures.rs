//! Hand computed cashflow fixtures for the financial contract proposal (#78).
//!
//! Each fixture states its instrument, units and expected cashflows as
//! literals computed by hand (shown in the comments), and then checks them
//! against an independent property rather than re-deriving the same formula:
//! counterparty conservation, equivalence between instrument forms, sign
//! symmetry or parity. Nothing here depends on crate types beyond `Decimal`,
//! so the fixtures stay valid while the contract types are still evolving.
//!
//! Conventions used throughout:
//! - prices and money are exact `Decimal`;
//! - timestamps are milliseconds since the Unix epoch (`u64`);
//! - quantities are positive and the side carries the sign;
//! - the only rounding step is explicit and named (`round_dp_with_strategy`
//!   half even, at the settlement asset scale);
//! - arithmetic that could overflow uses checked operations.

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
fn spot_round_trip_cashflows() {
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
fn spot_counterparties_conserve_value() {
    let (buyer_cash, seller_cash) = (dec!(-30030), dec!(29970));
    let (buyer_base, seller_base) = (dec!(0.5), dec!(-0.5));
    assert_eq!(buyer_base + seller_base, Decimal::ZERO);
    assert_eq!(buyer_cash + seller_cash, -dec!(60));
}

// ------------------------------------------------------ linear future

/// Linear (USDT margined) BTC future, multiplier 0.001 BTC per contract,
/// quote and settlement USDT.
///
/// Long 10 contracts @ 60000, close @ 61000: PnL 10 USDT.
/// Short the same: -10 USDT.
#[test]
fn linear_future_pnl_matches_spot_exposure() {
    let long_pnl = dec!(10);
    let short_pnl = dec!(-10);
    // Independent check: 10 contracts x 0.001 BTC is 0.01 BTC of exposure,
    // and a 0.01 BTC spot position gains the same over the same move.
    let exposure_btc = dec!(0.01);
    let spot_gain = exposure_btc * dec!(1000);
    assert_eq!(long_pnl, spot_gain);
    assert_eq!(long_pnl + short_pnl, Decimal::ZERO);
}

// ----------------------------------------------------- inverse future

/// Inverse (coin margined) BTC-USD future, contract value 100 USD, settled
/// in BTC. PnL in BTC is N x CV x (1/entry - 1/exit).
///
/// Long 10 @ 60000, exit 61000: 1000 x (1/60000 - 1/61000) = 1/3660 BTC =
/// 0.000273224043... BTC, settled at satoshi scale (half even) as
/// 0.00027322 BTC.
/// Long 10 @ 60000, exit 59000: 1000 x (1/60000 - 1/59000) = -1/3540 BTC =
/// -0.000282485875... BTC, settled as -0.00028249 BTC.
#[test]
fn inverse_future_settles_in_coin_at_satoshi_scale() {
    let up = dec!(0.00027322);
    let down = dec!(-0.00028249);

    // Independent check 1: the unrounded values are exact fractions;
    // dividing 1 by 3660 and 3540 and rounding once gives the settlements.
    let up_exact = Decimal::ONE.checked_div(dec!(3660)).unwrap();
    let down_exact = -Decimal::ONE.checked_div(dec!(3540)).unwrap();
    assert_eq!(round_btc(up_exact), up);
    assert_eq!(round_btc(down_exact), down);

    // Independent check 2: the USD value of the coin PnL at the exit price
    // equals the USD notional times the relative move against entry:
    // up: 1000 x (61000/60000 - 1) = 16.666... USD.
    let usd_at_exit = (up_exact * dec!(61000)).round_dp(6);
    assert_eq!(usd_at_exit, dec!(16.666667));

    // Convexity: an equal move up gains less coin than the move down loses.
    assert!(up.abs() < down.abs());
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
fn perpetual_funding_is_zero_sum() {
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
    // Funding moves cash between sides; it creates or destroys none: the
    // short's cash change is the negation of the long's.
    let short_total: Decimal = long_payments.iter().map(|p| -p).sum();
    assert_eq!(-long_total - short_total, Decimal::ZERO);
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
fn perpetual_funding_schedule_is_half_open() {
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
fn perpetual_funding_rounding_leaves_an_explicit_residual() {
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
fn option_expiry_cashflows_mirror_between_sides() {
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
fn option_payoffs_satisfy_parity_at_expiry() {
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
fn notional_overflow_is_detected() {
    assert_eq!(Decimal::MAX.checked_mul(dec!(2)), None);
    assert_eq!(dec!(60000).checked_mul(dec!(0.5)), Some(dec!(30000)));
}

/// Half even rounding at satoshi scale is unbiased on exact midpoints.
#[test]
fn settlement_rounding_is_half_even() {
    assert_eq!(round_btc(dec!(0.000000025)), dec!(0.00000002));
    assert_eq!(round_btc(dec!(0.000000035)), dec!(0.00000004));
}
