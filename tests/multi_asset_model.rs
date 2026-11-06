//! The closed-form joint quotes against independent references: the two by two
//! matrix root in closed form, the published single asset formula, and a direct
//! numerical solution of the paper's Hamilton-Jacobi equation (5).
//!
//! Every process here is synthetic. The two asset parameters follow section 6.1
//! of arXiv:1810.04383v5 in units of one trade (`gamma z = 0.08`, the paper's
//! `sigma` and `rho`), with exponential intensities in place of its logistic
//! ones, so no figure of the paper is reproduced as such.

#![cfg(feature = "multi-underlying")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::needless_range_loop
)]

use market_maker_rs::multi_underlying::multi_asset::{
    AssetQuotingParams, MAX_ASSETS, MultiAssetModel, QuotingObjective, RiskView,
};
use market_maker_rs::types::error::MMError;
use market_maker_rs::{Decimal, dec};
use rust_decimal::prelude::{FromPrimitive, ToPrimitive};

/// Sources of the numbers checked: never the code under test.
mod reference {
    /// The root of a two by two symmetric positive definite matrix:
    /// `sqrt(M) = (M + sqrt(det M) I) / sqrt(trace M + 2 sqrt(det M))`.
    pub fn root_2x2(m: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
        let det_root = (m[0][0] * m[1][1] - m[0][1] * m[1][0]).sqrt();
        let norm = (m[0][0] + m[1][1] + 2.0 * det_root).sqrt();
        [
            [(m[0][0] + det_root) / norm, m[0][1] / norm],
            [m[1][0] / norm, (m[1][1] + det_root) / norm],
        ]
    }

    /// Exponential intensities, unit trades and an objective.
    #[derive(Debug, Clone, Copy)]
    pub struct Problem {
        pub gamma: f64,
        pub sigma: [f64; 2],
        pub rho: f64,
        pub scale: f64,
        pub decay: f64,
        pub cara: bool,
    }

    impl Problem {
        fn constant(&self) -> f64 {
            if self.cara {
                let ratio = self.gamma / self.decay;
                (1.0 + ratio).powf(-(1.0 + 1.0 / ratio))
            } else {
                (-1.0_f64).exp()
            }
        }

        fn hamiltonian(&self, p: f64) -> f64 {
            self.scale / self.decay * self.constant() * (-self.decay * p).exp()
        }

        /// `delta*(p) - p`.
        pub fn base(&self) -> f64 {
            if self.cara {
                (1.0 + self.gamma / self.decay).ln() / self.gamma
            } else {
                1.0 / self.decay
            }
        }

        fn penalty(&self, q: [i32; 2]) -> f64 {
            let (a, b) = (
                f64::from(q[0]) * self.sigma[0],
                f64::from(q[1]) * self.sigma[1],
            );
            0.5 * self.gamma * (a * a + 2.0 * self.rho * a * b + b * b)
        }
    }

    /// Explicit backward Euler on equation (5) with terminal value zero.
    #[derive(Debug, Clone)]
    pub struct Solution {
        limit: i32,
        theta: Vec<f64>,
        problem: Problem,
    }

    impl Solution {
        pub fn solve(problem: Problem, limit: i32, horizon: f64, steps: usize) -> Self {
            let side = (2 * limit + 1) as usize;
            let index = |a: i32, b: i32| ((a + limit) as usize) * side + (b + limit) as usize;
            let dt = horizon / steps as f64;
            let mut theta = vec![0.0; side * side];
            let mut next = theta.clone();
            for _ in 0..steps {
                for a in -limit..=limit {
                    for b in -limit..=limit {
                        let here = theta[index(a, b)];
                        let mut flow = -problem.penalty([a, b]);
                        for (na, nb) in [(a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)] {
                            if na.abs() <= limit && nb.abs() <= limit {
                                flow += problem.hamiltonian(here - theta[index(na, nb)]);
                            }
                        }
                        next[index(a, b)] = here + dt * flow;
                    }
                }
                std::mem::swap(&mut theta, &mut next);
            }
            Self {
                limit,
                theta,
                problem,
            }
        }

        fn value(&self, a: i32, b: i32) -> f64 {
            let side = (2 * self.limit + 1) as usize;
            self.theta[((a + self.limit) as usize) * side + (b + self.limit) as usize]
        }

        /// The optimal bid distance of Theorems 1 and 2 at inventory `q`.
        pub fn bid(&self, asset: usize, q: [i32; 2]) -> f64 {
            let [a, b] = q;
            let up = if asset == 0 { (a + 1, b) } else { (a, b + 1) };
            self.value(a, b) - self.value(up.0, up.1) + self.problem.base()
        }

        /// The optimal ask distance of Theorems 1 and 2 at inventory `q`.
        pub fn ask(&self, asset: usize, q: [i32; 2]) -> f64 {
            let [a, b] = q;
            let down = if asset == 0 { (a - 1, b) } else { (a, b - 1) };
            self.value(a, b) - self.value(down.0, down.1) + self.problem.base()
        }
    }
}

fn d(value: f64) -> Decimal {
    Decimal::from_f64(value).unwrap()
}

fn f(value: Decimal) -> f64 {
    value.to_f64().unwrap()
}

fn asset(sigma: Decimal, scale: Decimal, decay: Decimal, size: Decimal) -> AssetQuotingParams {
    AssetQuotingParams::new(sigma, scale, decay, size, dec!(1000)).unwrap()
}

/// The two assets of section 6.1 in units of one trade.
fn paper_assets() -> Vec<AssetQuotingParams> {
    vec![
        asset(dec!(1.2), dec!(10), dec!(20), dec!(1)),
        asset(dec!(0.6), dec!(10), dec!(20), dec!(1)),
    ]
}

fn correlation(rho: Decimal) -> Vec<Vec<Decimal>> {
    vec![vec![dec!(1), rho], vec![rho, dec!(1)]]
}

fn paper_model(rho: Decimal, objective: QuotingObjective, view: RiskView) -> MultiAssetModel {
    MultiAssetModel::new(
        paper_assets(),
        &correlation(rho),
        dec!(0.08),
        objective,
        view,
    )
    .unwrap()
}

fn correlated(rho: Decimal) -> MultiAssetModel {
    paper_model(
        rho,
        QuotingObjective::RiskAdjustedExpectation,
        RiskView::Correlated,
    )
}

fn diagonal(rho: Decimal) -> MultiAssetModel {
    paper_model(
        rho,
        QuotingObjective::RiskAdjustedExpectation,
        RiskView::Diagonal,
    )
}

/// `Gamma` by the closed form, from the definition of the paper.
fn reference_gamma(depths: [f64; 2], covariance: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    let scaled = [
        [
            depths[0] * covariance[0][0],
            (depths[0] * depths[1]).sqrt() * covariance[0][1],
        ],
        [
            (depths[0] * depths[1]).sqrt() * covariance[1][0],
            depths[1] * covariance[1][1],
        ],
    ];
    let root = reference::root_2x2(scaled);
    let mut gamma = [[0.0; 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            gamma[i][j] = root[i][j] / (depths[i] * depths[j]).sqrt();
        }
    }
    gamma
}

#[test]
fn test_risk_matrix_two_assets_model_b_matches_closed_root() {
    let model = correlated(dec!(0.5));
    let e = (-1.0_f64).exp();
    let depth = 2.0 * 10.0 * e * 20.0;
    let expected = reference_gamma([depth, depth], [[1.44, 0.36], [0.36, 0.36]]);
    for i in 0..2 {
        for j in 0..2 {
            let got = f(model.risk_matrix(i, j).unwrap());
            assert!(
                (got - expected[i][j]).abs() < 1e-12,
                "({i}, {j}): {got} against {}",
                expected[i][j]
            );
        }
    }
}

#[test]
fn test_risk_matrix_two_assets_model_a_uses_the_utility_constant() {
    let model = paper_model(
        dec!(0.5),
        QuotingObjective::ExponentialUtility,
        RiskView::Correlated,
    );
    let ratio = 0.08_f64 / 20.0;
    let constant = (1.0 + ratio).powf(-(1.0 + 1.0 / ratio));
    let depth = 2.0 * 10.0 * constant * 20.0;
    let expected = reference_gamma([depth, depth], [[1.44, 0.36], [0.36, 0.36]]);
    for i in 0..2 {
        for j in 0..2 {
            let got = f(model.risk_matrix(i, j).unwrap());
            assert!((got - expected[i][j]).abs() < 1e-12);
        }
    }
}

#[test]
fn test_risk_matrix_depends_on_each_assets_liquidity_and_size() {
    // Different scale, decay and size per asset: the depth of each matters.
    let assets = vec![
        asset(dec!(1), dec!(5), dec!(2), dec!(3)),
        asset(dec!(2), dec!(7), dec!(11), dec!(0.5)),
    ];
    let model = MultiAssetModel::new(
        assets,
        &correlation(dec!(-0.4)),
        dec!(0.2),
        QuotingObjective::RiskAdjustedExpectation,
        RiskView::Correlated,
    )
    .unwrap();
    let e = (-1.0_f64).exp();
    let depths = [2.0 * 5.0 * e * 2.0 * 3.0, 2.0 * 7.0 * e * 11.0 * 0.5];
    let expected = reference_gamma(depths, [[1.0, -0.8], [-0.8, 4.0]]);
    for i in 0..2 {
        for j in 0..2 {
            assert!((f(model.risk_matrix(i, j).unwrap()) - expected[i][j]).abs() < 1e-12);
        }
    }
}

#[test]
fn test_risk_matrix_five_assets_solves_defining_equation() {
    // `Gamma D+ Gamma = Sigma`, with a correlation no closed root covers.
    let sigmas = [1.0, 0.5, 2.0, 0.25, 1.5];
    let rho = [
        [1.0, 0.3, -0.2, 0.1, 0.4],
        [0.3, 1.0, 0.25, -0.15, 0.2],
        [-0.2, 0.25, 1.0, 0.35, -0.3],
        [0.1, -0.15, 0.35, 1.0, 0.05],
        [0.4, 0.2, -0.3, 0.05, 1.0],
    ];
    let assets: Vec<_> = sigmas
        .iter()
        .map(|sigma| asset(d(*sigma), dec!(10), dec!(4), dec!(1)))
        .collect();
    let matrix: Vec<Vec<Decimal>> = rho
        .iter()
        .map(|r| r.iter().map(|v| d(*v)).collect())
        .collect();
    let model = MultiAssetModel::new(
        assets,
        &matrix,
        dec!(0.1),
        QuotingObjective::RiskAdjustedExpectation,
        RiskView::Correlated,
    )
    .unwrap();
    let depth = 2.0 * 10.0 * (-1.0_f64).exp() * 4.0;
    for i in 0..5 {
        for j in 0..5 {
            let rebuilt: f64 = (0..5)
                .map(|k| {
                    f(model.risk_matrix(i, k).unwrap())
                        * depth
                        * f(model.risk_matrix(k, j).unwrap())
                })
                .sum();
            let sigma = rho[i][j] * sigmas[i] * sigmas[j];
            assert!((rebuilt - sigma).abs() < 1e-10, "({i}, {j})");
            assert_eq!(model.risk_matrix(i, j), model.risk_matrix(j, i));
        }
    }
}

#[test]
fn test_offsets_one_asset_model_a_reduce_to_the_single_asset_closed_form() {
    // Gueant, Lehalle and Fernandez-Tapia, as the paper says it must coincide
    // with one asset and unit size: slope sqrt(sigma^2 gamma / (2 k A)
    // (1 + gamma / k)^(1 + k / gamma)), base ln(1 + gamma / k) / gamma.
    let (sigma, a, k, gamma) = (0.3_f64, 140.0_f64, 1.5_f64, 0.1_f64);
    let model = MultiAssetModel::new(
        vec![asset(d(sigma), d(a), d(k), dec!(1))],
        &[vec![dec!(1)]],
        d(gamma),
        QuotingObjective::ExponentialUtility,
        RiskView::Correlated,
    )
    .unwrap();
    let slope =
        (sigma * sigma * gamma / (2.0 * k * a) * (1.0 + gamma / k).powf(1.0 + k / gamma)).sqrt();
    let base = (1.0 + gamma / k).ln() / gamma;
    for q in [-3.0, 0.0, 2.0] {
        let offsets = model.offsets(0, &[d(q)]).unwrap();
        let bid = base + (q + 0.5) * slope;
        let ask = base + (0.5 - q) * slope;
        assert!((f(offsets.bid) - bid).abs() < 1e-12, "bid at {q}");
        assert!((f(offsets.ask) - ask).abs() < 1e-12, "ask at {q}");
    }
}

#[test]
fn test_offsets_two_assets_follow_the_affine_formula() {
    let model = correlated(dec!(0.5));
    let q = [dec!(3), dec!(-2)];
    let gamma_root = 0.08_f64.sqrt();
    for i in 0..2 {
        let row: f64 = (0..2)
            .map(|j| f(q[j]) * f(model.risk_matrix(i, j).unwrap()))
            .sum();
        let own = f(model.risk_matrix(i, i).unwrap());
        let offsets = model.offsets(i, &q).unwrap();
        let bid = gamma_root * (row + 0.5 * own) + 1.0 / 20.0;
        let ask = -gamma_root * (row - 0.5 * own) + 1.0 / 20.0;
        assert!((f(offsets.bid) - bid).abs() < 1e-12);
        assert!((f(offsets.ask) - ask).abs() < 1e-12);
    }
}

#[test]
fn test_model_a_with_vanishing_risk_aversion_tends_to_model_b_constant() {
    // `1 + gamma z / k` rounds to one in `f64` here, and the constant must still
    // be near `1/e`, so that Gamma keeps the depth of Model B.
    let build = |gamma, objective| {
        MultiAssetModel::new(
            vec![asset(dec!(1), dec!(10), dec!(20), dec!(1))],
            &[vec![dec!(1)]],
            gamma,
            objective,
            RiskView::Correlated,
        )
        .unwrap()
    };
    let tiny = build(
        dec!(0.00000000000000000001),
        QuotingObjective::ExponentialUtility,
    );
    let limit = build(dec!(0.08), QuotingObjective::RiskAdjustedExpectation);
    let (a, b) = (
        f(tiny.risk_matrix(0, 0).unwrap()),
        f(limit.risk_matrix(0, 0).unwrap()),
    );
    assert!((a - b).abs() < 1e-12, "{a} against {b}");
    // The base term is 1 / k and the risk buffer `sqrt(gamma) Gamma / 2` is 4e-12.
    assert!((f(tiny.half_spread(0).unwrap()) - 1.0 / 20.0).abs() < 1e-11);
}

#[test]
fn test_model_a_constant_is_stable_where_the_direct_power_loses_it() {
    // `(1 + r)^-(1 + 1/r)` read 1.0567 and 0.6065 of the limit at 1e-15 and 1e-16.
    for ratio in [
        dec!(0.00000001),
        dec!(0.000000000000001),
        dec!(0.0000000000000001),
    ] {
        let build = |objective| {
            MultiAssetModel::new(
                vec![asset(dec!(1), dec!(10), dec!(1), dec!(1))],
                &[vec![dec!(1)]],
                ratio,
                objective,
                RiskView::Correlated,
            )
            .unwrap()
        };
        let a = f(build(QuotingObjective::ExponentialUtility)
            .risk_matrix(0, 0)
            .unwrap());
        let b = f(build(QuotingObjective::RiskAdjustedExpectation)
            .risk_matrix(0, 0)
            .unwrap());
        assert!((a / b - 1.0).abs() < 1e-7, "ratio {ratio}: {}", a / b);
    }
}

#[test]
fn test_half_spread_follows_the_trade_size_in_both_models() {
    let (sigma, a, k, gamma, z) = (1.2_f64, 10.0_f64, 20.0_f64, 0.08_f64, 3.0_f64);
    let ratio = gamma * z / k;
    for (objective, constant, base) in [
        (
            QuotingObjective::RiskAdjustedExpectation,
            (-1.0_f64).exp(),
            1.0 / k,
        ),
        (
            QuotingObjective::ExponentialUtility,
            (1.0 + ratio).powf(-(1.0 + 1.0 / ratio)),
            (1.0 + ratio).ln() / (gamma * z),
        ),
    ] {
        let model = MultiAssetModel::new(
            vec![asset(d(sigma), d(a), d(k), d(z))],
            &[vec![dec!(1)]],
            d(gamma),
            objective,
            RiskView::Correlated,
        )
        .unwrap();
        // With one asset, Gamma = sigma / sqrt(D+).
        let risk = sigma / (2.0 * a * constant * k * z).sqrt();
        let half = 0.5 * gamma.sqrt() * z * risk + base;
        let got = f(model.half_spread(0).unwrap());
        assert!(
            (got - half).abs() < 1e-12,
            "{objective:?}: {got} against {half}"
        );
    }
}

#[test]
fn test_half_spread_does_not_depend_on_inventory() {
    let model = correlated(dec!(0.5));
    let flat = model.offsets(0, &[dec!(0), dec!(0)]).unwrap();
    let loaded = model.offsets(0, &[dec!(7), dec!(-4)]).unwrap();
    let half = model.half_spread(0).unwrap();
    assert_eq!(flat.bid, half);
    assert_eq!(flat.ask, half);
    let sum = loaded.bid.checked_add(loaded.ask).unwrap();
    assert!((sum - half - half).abs() < dec!(0.000000000001));
}

#[test]
fn test_offsets_own_inventory_moves_both_quotes_down() {
    let model = correlated(dec!(0.5));
    let flat = model.offsets(0, &[dec!(0), dec!(0)]).unwrap();
    let long = model.offsets(0, &[dec!(3), dec!(0)]).unwrap();
    let short = model.offsets(0, &[dec!(-3), dec!(0)]).unwrap();
    assert!(long.bid > flat.bid, "long bids lower");
    assert!(long.ask < flat.ask, "long asks lower");
    assert!(short.bid < flat.bid, "short bids higher");
    assert!(short.ask > flat.ask, "short asks higher");
}

#[test]
fn test_skew_hedged_portfolio_below_diagonal_and_concentrated_above() {
    let rho = dec!(0.5);
    let (joint, separate) = (correlated(rho), diagonal(rho));
    let hedged = [dec!(4), dec!(-4)];
    let concentrated = [dec!(4), dec!(4)];
    let diagonal_skew = separate.skew(0, &hedged).unwrap();
    // The diagonal view cannot see the other leg.
    assert_eq!(diagonal_skew, separate.skew(0, &concentrated).unwrap());
    assert!(joint.skew(0, &hedged).unwrap() < joint.skew(0, &concentrated).unwrap());
    assert!(joint.skew(0, &concentrated).unwrap() > diagonal_skew);
    // The same two portfolios swap roles when the correlation is negative.
    let inverse = correlated(dec!(-0.5));
    assert!(inverse.skew(0, &hedged).unwrap() > inverse.skew(0, &concentrated).unwrap());
    // Portfolio variance orders them the same way.
    assert!(
        joint.portfolio_variance(&hedged).unwrap()
            < joint.portfolio_variance(&concentrated).unwrap()
    );
    assert_eq!(
        separate.portfolio_variance(&hedged).unwrap(),
        separate.portfolio_variance(&concentrated).unwrap()
    );
}

#[test]
fn test_skew_hedged_leg_of_low_volatility_asset_is_pulled_below_diagonal() {
    // Asset 2 is the smaller risk: its own skew must shrink too when it hedges.
    let rho = dec!(0.5);
    let hedged = [dec!(4), dec!(-4)];
    let joint = correlated(rho).skew(1, &hedged).unwrap();
    let separate = diagonal(rho).skew(1, &hedged).unwrap();
    assert!(joint > separate, "{joint} against {separate}");
}

#[test]
fn test_diagonal_view_equals_correlated_view_without_correlation() {
    let (joint, separate) = (correlated(dec!(0)), diagonal(dec!(0)));
    let q = [dec!(5), dec!(-3)];
    for i in 0..2 {
        let (a, b) = (
            joint.offsets(i, &q).unwrap(),
            separate.offsets(i, &q).unwrap(),
        );
        assert!((a.bid - b.bid).abs() < dec!(0.000000000001));
        assert!((a.ask - b.ask).abs() < dec!(0.000000000001));
    }
}

#[test]
fn test_diagonal_view_each_asset_equals_its_own_one_asset_model() {
    let rho = dec!(0.7);
    let two = diagonal(rho);
    for (i, params) in paper_assets().into_iter().enumerate() {
        let one = MultiAssetModel::new(
            vec![params],
            &[vec![dec!(1)]],
            dec!(0.08),
            QuotingObjective::RiskAdjustedExpectation,
            RiskView::Correlated,
        )
        .unwrap();
        let q = [dec!(9), dec!(-6)];
        let a = two.offsets(i, &q).unwrap();
        let b = one.offsets(0, &[q[i]]).unwrap();
        assert!((a.bid - b.bid).abs() < dec!(0.000000000001));
        assert!((a.ask - b.ask).abs() < dec!(0.000000000001));
    }
}

#[test]
fn test_perfect_correlation_is_accepted_as_a_singular_matrix() {
    let model = correlated(dec!(1));
    let entry = f(model.risk_matrix(0, 1).unwrap());
    assert!(entry > 0.0);
    // Equal and opposite risk units cancel: sigma_1 q_1 + sigma_2 q_2 = 0.
    let flat = model.portfolio_variance(&[dec!(1), dec!(-2)]).unwrap();
    assert!(flat.abs() < dec!(0.000000001));
}

#[test]
fn test_model_invalid_correlation_is_refused() {
    let new = |matrix: &[Vec<Decimal>]| {
        MultiAssetModel::new(
            paper_assets(),
            matrix,
            dec!(0.08),
            QuotingObjective::RiskAdjustedExpectation,
            RiskView::Correlated,
        )
    };
    for (name, matrix) in [
        (
            "not symmetric",
            vec![vec![dec!(1), dec!(0.5)], vec![dec!(0.4), dec!(1)]],
        ),
        (
            "diagonal not one",
            vec![vec![dec!(0.9), dec!(0.5)], vec![dec!(0.5), dec!(1)]],
        ),
        (
            "outside the range",
            vec![vec![dec!(1), dec!(1.1)], vec![dec!(1.1), dec!(1)]],
        ),
        ("wrong size", vec![vec![dec!(1)]]),
        ("not square", vec![vec![dec!(1), dec!(0)], vec![dec!(0)]]),
    ] {
        let error = new(&matrix).unwrap_err();
        assert!(matches!(error, MMError::InvalidConfiguration(_)), "{name}");
    }
}

#[test]
fn test_model_correlation_not_semidefinite_is_refused_in_both_views() {
    // Each pair is valid; together they are impossible.
    let matrix = vec![
        vec![dec!(1), dec!(0.9), dec!(0.9)],
        vec![dec!(0.9), dec!(1), dec!(-0.9)],
        vec![dec!(0.9), dec!(-0.9), dec!(1)],
    ];
    for view in [RiskView::Correlated, RiskView::Diagonal] {
        let assets = vec![asset(dec!(1), dec!(10), dec!(4), dec!(1)); 3];
        let error = MultiAssetModel::new(
            assets,
            &matrix,
            dec!(0.1),
            QuotingObjective::RiskAdjustedExpectation,
            view,
        )
        .unwrap_err();
        assert!(
            matches!(error, MMError::InvalidConfiguration(_)),
            "{view:?}"
        );
    }
}

#[test]
fn test_model_invalid_sizes_and_risk_aversion_are_refused() {
    let new = |assets, gamma| {
        MultiAssetModel::new(
            assets,
            &[vec![dec!(1)]],
            gamma,
            QuotingObjective::RiskAdjustedExpectation,
            RiskView::Correlated,
        )
    };
    let one = vec![asset(dec!(1), dec!(10), dec!(4), dec!(1))];
    assert!(matches!(
        new(vec![], dec!(0.1)),
        Err(MMError::InvalidConfiguration(_))
    ));
    assert!(matches!(
        new(one.clone(), dec!(0)),
        Err(MMError::InvalidConfiguration(_))
    ));
    assert!(matches!(
        new(one, dec!(-0.1)),
        Err(MMError::InvalidConfiguration(_))
    ));
    let too_many = vec![asset(dec!(1), dec!(10), dec!(4), dec!(1)); MAX_ASSETS + 1];
    let identity: Vec<Vec<Decimal>> = (0..=MAX_ASSETS)
        .map(|i| {
            (0..=MAX_ASSETS)
                .map(|j| if i == j { dec!(1) } else { dec!(0) })
                .collect()
        })
        .collect();
    let error = MultiAssetModel::new(
        too_many,
        &identity,
        dec!(0.1),
        QuotingObjective::RiskAdjustedExpectation,
        RiskView::Correlated,
    )
    .unwrap_err();
    assert!(matches!(error, MMError::InvalidConfiguration(_)));
}

#[test]
fn test_asset_params_not_positive_are_refused() {
    let one = dec!(1);
    for (sigma, scale, decay, size, limit) in [
        (dec!(0), one, one, one, one),
        (one, dec!(0), one, one, one),
        (one, one, dec!(-1), one, one),
        (one, one, one, dec!(0), one),
        (one, one, one, one, dec!(0)),
        // A trade above the limit could never quote; one at the limit can.
        (one, one, one, dec!(2), one),
    ] {
        assert!(matches!(
            AssetQuotingParams::new(sigma, scale, decay, size, limit),
            Err(MMError::InvalidConfiguration(_))
        ));
    }
    assert!(AssetQuotingParams::new(one, one, one, one, one).is_ok());
}

#[test]
fn test_offsets_bad_inputs_are_errors() {
    let model = correlated(dec!(0.5));
    assert!(matches!(
        model.offsets(0, &[dec!(1)]),
        Err(MMError::InvalidPositionUpdate(_))
    ));
    assert!(matches!(
        model.offsets(2, &[dec!(1), dec!(1)]),
        Err(MMError::InvalidConfiguration(_))
    ));
    assert!(matches!(
        model.portfolio_variance(&[dec!(1)]),
        Err(MMError::InvalidPositionUpdate(_))
    ));
    assert_eq!(model.half_spread(2), None);
    assert_eq!(model.risk_matrix(2, 0), None);
    assert_eq!(model.covariance(0, 2), None);
}

#[test]
fn test_offsets_inventory_beyond_decimal_range_is_an_error() {
    let wild = MultiAssetModel::new(
        vec![asset(dec!(1000000), dec!(1), dec!(1), dec!(1))],
        &[vec![dec!(1)]],
        dec!(1),
        QuotingObjective::RiskAdjustedExpectation,
        RiskView::Correlated,
    )
    .unwrap();
    assert!(matches!(
        wild.offsets(0, &[Decimal::MAX]),
        Err(MMError::NumericalError(_))
    ));
    assert!(matches!(
        wild.portfolio_variance(&[Decimal::MAX]),
        Err(MMError::NumericalError(_))
    ));
}

#[test]
fn test_model_values_beyond_decimal_range_are_refused() {
    // The covariance of this volatility does not fit in a `Decimal`.
    let error = MultiAssetModel::new(
        vec![asset(
            dec!(100000000000000000000),
            dec!(1),
            dec!(1),
            dec!(1),
        )],
        &[vec![dec!(1)]],
        dec!(1),
        QuotingObjective::RiskAdjustedExpectation,
        RiskView::Correlated,
    )
    .unwrap_err();
    assert!(matches!(error, MMError::NumericalError(_)));
}

/// The two models of section 6.1, in units of one trade.
fn problem(rho: f64, cara: bool) -> reference::Problem {
    reference::Problem {
        gamma: 0.08,
        sigma: [1.2, 0.6],
        rho,
        scale: 10.0,
        decay: 20.0,
        cara,
    }
}

/// Worst absolute gap between the model and the solved equation, over the bids
/// and asks of both assets on inventories up to `reach` in each asset.
fn worst_gap(model: &MultiAssetModel, solution: &reference::Solution, reach: i32) -> f64 {
    let mut worst = 0.0_f64;
    for a in -reach..=reach {
        for b in -reach..=reach {
            let q = [d(f64::from(a)), d(f64::from(b))];
            for i in 0..2 {
                let offsets = model.offsets(i, &q).unwrap();
                worst = worst
                    .max((f(offsets.bid) - solution.bid(i, [a, b])).abs())
                    .max((f(offsets.ask) - solution.ask(i, [a, b])).abs());
            }
        }
    }
    worst
}

#[test]
fn test_numerical_equation_correlated_model_tracks_cross_inventory_effect() {
    for (rho_value, rho) in [(0.5, dec!(0.5)), (-0.5, dec!(-0.5))] {
        for (cara, objective) in [
            (false, QuotingObjective::RiskAdjustedExpectation),
            (true, QuotingObjective::ExponentialUtility),
        ] {
            let solution = reference::Solution::solve(problem(rho_value, cara), 8, 5.0, 10_000);
            let joint = paper_model(rho, objective, RiskView::Correlated);
            let separate = paper_model(rho, objective, RiskView::Diagonal);
            // Slope of the bid of asset 1 in the inventory of asset 2.
            let solved = (solution.bid(0, [0, 1]) - solution.bid(0, [0, -1])) / 2.0;
            let closed = 0.08_f64.sqrt() * f(joint.risk_matrix(0, 1).unwrap());
            let gap = (
                worst_gap(&joint, &solution, 2),
                worst_gap(&separate, &solution, 2),
            );
            assert_eq!(solved.signum(), rho_value.signum());
            // Measured: 9 % on the slope, gaps of 0.0032 and 0.0119 against a
            // half spread of 0.053. The bounds leave room, not slack to hide a
            // lost cross term, which would make the slope error 100 %.
            assert!(
                (solved - closed).abs() < 0.2 * solved.abs(),
                "{solved} {closed}"
            );
            assert!(gap.0 < 0.005, "{gap:?}");
            assert!(gap.0 < 0.5 * gap.1, "{gap:?}");
        }
    }
}

#[test]
fn test_a_supplied_correlation_is_held_to_the_floor_of_rounding_alone() {
    // Two rows, so a floor of -2e-12 for the eigenvalue `1 - rho`. A correlation the crate
    // derives may be clipped past it; one supplied is not.
    let new = |rho: Decimal| {
        MultiAssetModel::new(
            paper_assets(),
            &correlation(rho),
            dec!(0.08),
            QuotingObjective::RiskAdjustedExpectation,
            RiskView::Correlated,
        )
    };
    assert!(new(dec!(1.000000000001)).is_ok());
    let error = new(dec!(1.000000000003)).unwrap_err();
    let MMError::InvalidConfiguration(message) = &error else {
        panic!("{error:?}");
    };
    assert!(
        message.contains("eigenvalue -0.0000000000029")
            && message.contains("floor -0.000000000002"),
        "{message}"
    );
}
