//! The stratified stationary bootstrap: groups resampled on their own streams, pooled by the
//! unweighted mean of their replicate means.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    missing_docs
)]

use market_maker_rs::Decimal;
use market_maker_rs::backtest::evaluation::{
    BootstrapConfig, BootstrapOutcome, block_bootstrap, stratified_block_bootstrap,
};
use market_maker_rs::dec;

fn config(seed: u64) -> BootstrapConfig {
    BootstrapConfig {
        replicates: 2000,
        mean_block_days: 2,
        confidence_percent: 90,
        seed,
        min_days: 4,
    }
}

fn interval(outcome: BootstrapOutcome) -> (f64, f64, f64, u32) {
    match outcome {
        BootstrapOutcome::Interval {
            mean,
            lower,
            upper,
            days,
        } => (mean, lower, upper, days),
        other => panic!("no interval: {other:?}"),
    }
}

fn calm() -> Vec<Decimal> {
    vec![dec!(1.5), dec!(-2), dec!(0.25), dec!(3), dec!(-0.75)]
}

fn thin() -> Vec<Decimal> {
    vec![dec!(-4), dec!(2.5), dec!(1), dec!(0), dec!(6)]
}

#[test]
fn test_a_single_group_is_exactly_the_block_bootstrap() {
    let series = calm();
    let single = interval(stratified_block_bootstrap(&[&series], &config(280)).unwrap());
    let plain = interval(block_bootstrap(&series, &config(280)).unwrap());
    assert_eq!(single, plain);
}

#[test]
fn test_the_pool_is_the_unweighted_mean_of_group_means_whatever_their_lengths() {
    // A long group does not weigh more: calm (5 days, mean 0.4) and a 10 day group of 2.
    let long = vec![dec!(2); 10];
    let (mean, lower, upper, days) =
        interval(stratified_block_bootstrap(&[&calm(), &long], &config(280)).unwrap());
    assert!((mean - (0.4 + 2.0) / 2.0).abs() < 1e-12, "{mean}");
    assert_eq!(days, 15);
    // A constant group contributes its value to every replicate.
    let calm_alone = interval(block_bootstrap(&calm(), &config(280)).unwrap());
    assert!((lower - (calm_alone.1 + 2.0) / 2.0).abs() < 1e-12);
    assert!((upper - (calm_alone.2 + 2.0) / 2.0).abs() < 1e-12);
}

#[test]
fn test_each_group_draws_from_its_own_stream() {
    // The same values in two groups: with one shared seed both would draw the same days, and
    // their replicate means, and so the pool's spread, would equal one group's. With seeds
    // seed and seed + 1 the draws differ, and averaging two different draws narrows the
    // interval.
    let series = calm();
    let one = interval(block_bootstrap(&series, &config(280)).unwrap());
    let both = interval(stratified_block_bootstrap(&[&series, &series], &config(280)).unwrap());
    assert_eq!(both.0, one.0);
    assert!(both.2 - both.1 < one.2 - one.1, "{both:?} against {one:?}");
    // Group 1 is resampled at seed + 1: behind a constant group 0, the pool's bounds are the
    // second group's own bounds at seed 281, halved and shifted by the constant.
    let constant = vec![dec!(2); 5];
    let pooled = interval(stratified_block_bootstrap(&[&constant, &thin()], &config(280)).unwrap());
    let at_281 = interval(block_bootstrap(&thin(), &config(281)).unwrap());
    let at_280 = interval(block_bootstrap(&thin(), &config(280)).unwrap());
    assert!(
        (pooled.1 - (2.0 + at_281.1) / 2.0).abs() < 1e-12,
        "{pooled:?} {at_281:?}"
    );
    assert!((pooled.2 - (2.0 + at_281.2) / 2.0).abs() < 1e-12);
    assert!(at_281.1 != at_280.1 || at_281.2 != at_280.2);
}

#[test]
fn test_the_same_inputs_give_the_same_interval_and_another_seed_another() {
    let groups: [&[Decimal]; 2] = [&calm(), &thin()];
    let a = stratified_block_bootstrap(&groups, &config(280)).unwrap();
    let b = stratified_block_bootstrap(&groups, &config(280)).unwrap();
    assert_eq!(a, b);
    let c = interval(stratified_block_bootstrap(&groups, &config(9)).unwrap());
    let a = interval(a);
    assert_eq!(a.0, c.0);
    assert!(a.1 != c.1 || a.2 != c.2);
}

#[test]
fn test_no_group_an_empty_group_an_invalid_configuration_or_an_overflowing_seed_is_an_error() {
    assert!(stratified_block_bootstrap(&[], &config(280)).is_err());
    let empty: &[Decimal] = &[];
    let error = stratified_block_bootstrap(&[&calm(), empty], &config(280)).unwrap_err();
    assert!(error.to_string().contains("group 1 has no days"), "{error}");
    let mut bad = config(280);
    bad.replicates = 0;
    assert!(stratified_block_bootstrap(&[&calm()], &bad).is_err());
    let mut bad = config(280);
    bad.confidence_percent = 100;
    assert!(stratified_block_bootstrap(&[&calm()], &bad).is_err());
    // Group 0 keeps u64::MAX; group 1 would overflow it.
    assert!(stratified_block_bootstrap(&[&calm()], &config(u64::MAX)).is_ok());
    let error = stratified_block_bootstrap(&[&calm(), &thin()], &config(u64::MAX)).unwrap_err();
    assert!(error.to_string().contains("overflows"), "{error}");
}

#[test]
fn test_a_group_shorter_than_the_minimum_gives_no_interval() {
    let short = vec![dec!(1), dec!(2), dec!(3)];
    assert_eq!(
        stratified_block_bootstrap(&[&calm(), &short], &config(280)).unwrap(),
        BootstrapOutcome::Insufficient {
            days: 3,
            required: 4
        }
    );
}
